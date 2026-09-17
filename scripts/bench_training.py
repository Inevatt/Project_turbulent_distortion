"""Benchmark real training; measure loader startup once per estimated epoch.

Default: drain the prefetch queue, then time training and a full validation pass.
No checkpoint is written; model, physics and experiment config are unchanged.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from src.data import split_indices
from src.distortion import LEVELS
from src.metrics import psnr, ssim
from src.pair_pipeline import pairs_class, render_batch
from src.physics_version import physics_id
from src.unet import UNet

BENCHMARK_VERSION = 4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/seed1.yaml')
    p.add_argument('--level', default='d3', choices=sorted(LEVELS))
    p.add_argument('--steps', type=int, default=None,
                   help='measured train batches; default max(64, 4 * workers)')
    p.add_argument('--warmup', type=int, default=None,
                   help='warmup train batches; default max(5, 2 * workers)')
    p.add_argument('--val-steps', type=int, default=0,
                   help='0 = complete validation pass (recommended)')
    p.add_argument('--full-epoch', action='store_true',
                   help='time all train and validation batches; keep configured LR budget')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.full_epoch and (a.steps is not None or a.warmup is not None or a.val_steps != 0):
        p.error('--full-epoch cannot be combined with step-count overrides')
    cfg = yaml.safe_load(Path(a.config).read_text())
    print(f'benchmark_version={BENCHMARK_VERSION}', flush=True)
    d, t, m = cfg['data'], cfg['train'], cfg['model']
    workers, batch = int(t['num_workers']), int(t['batch_size'])
    warmup = max(5, 2 * workers) if a.warmup is None else a.warmup
    steps = max(64, 4 * workers) if a.steps is None else a.steps
    if steps < 1 or warmup < 0 or a.val_steps < 0:
        p.error('invalid step count')
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; run on the training server')
    torch.set_num_threads(1)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    deg = LEVELS[a.level](cfg['degradation']['diffraction_fwhm_px'])
    tr, va, te = split_indices(d['tiles'], d['val_frac'], d['test_frac'], cfg['split_seed'])
    kw = dict(tiles_dir=d['tiles'], degradation=deg, margin_px=d['margin_px'],
              crop_px=d['crop_px'], d_over_r0_range=d['d_over_r0_range'])
    ds = pairs_class(deg)(indices=tr, seed=cfg['seed'],
                          samples_per_tile=d['samples_per_tile'], **kw)
    vd = pairs_class(deg, True)(indices=va, seed=cfg['eval_seed'], **kw)
    if hasattr(deg, 'sampler'):
        deg.sampler._prepare(d['crop_px'] + 2 * d['margin_px'])
    dlkw = dict(batch_size=batch, num_workers=workers, pin_memory=device.type == 'cuda',
                **({'prefetch_factor': 1} if workers else {}))
    gen = torch.Generator().manual_seed(cfg['seed'])
    dl = DataLoader(ds, shuffle=True, generator=gen, drop_last=True, **dlkw)
    vl = DataLoader(vd, shuffle=False, **dlkw)
    if a.full_epoch:
        warmup, steps = 0, len(dl)
    val_steps = a.val_steps or len(vl)
    if steps < 1 or len(dl) < steps + warmup or not val_steps or len(vl) < val_steps:
        raise SystemExit('not enough batches for requested benchmark')
    source = np.load(Path(d['tiles']) / 'source_id.npy')
    counts = {
        name: dict(tiles=len(part), sources=int(len(np.unique(source[part]))))
        for name, part in [('train', tr), ('val', va), ('test', te)]
    }
    print('dataset: ' + json.dumps(counts), flush=True)
    print(f'train warmup={warmup}, measured={steps}; validation={val_steps}/{len(vl)}', flush=True)
    torch.manual_seed(cfg['seed'])
    net = UNet(in_ch=m['channels'], out_ch=m['channels'], base=m['base'], depth=m['depth']).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(t['lr']))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dl) * int(t['epochs']))
    loss_fn = torch.nn.L1Loss()
    ds.set_epoch(0)
    gen.manual_seed(cfg['seed'])
    net.train()

    # Event markers measure the CUDA stream without inserting a barrier
    # between synthesis and the network. Read them only after the ordinary
    # loss.item()/metric CPU transfer has already completed that work.
    render_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if device.type == 'cuda' else None

    def timed_render(packed):
        start = time.perf_counter()
        if render_events:
            render_events[0].record()
        result = render_batch(packed, deg, device)
        if render_events:
            render_events[1].record()
        return result, time.perf_counter() - start

    def render_duration(cpu_seconds):
        if render_events:
            return render_events[0].elapsed_time(render_events[1]) / 1000.
        return cpu_seconds

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    sync()
    epoch_start = time.perf_counter()
    start = time.perf_counter()
    it = iter(dl)
    train_iterator_seconds = time.perf_counter() - start
    timing, warmup_times = [], []
    loss_sum, seen = 0., 0
    for i in range(warmup + steps):
        start = time.perf_counter()
        packed = next(it)
        loaded = time.perf_counter()
        (x, y, _), render_cpu = timed_render(packed)
        loss = loss_fn(net(x), y)
        if not torch.isfinite(loss):
            raise SystemExit('non-finite training loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        loss_value = loss.item()  # same synchronization/aggregation as train.py
        loss_sum += loss_value * len(x)
        seen += len(x)
        elapsed = time.perf_counter() - start
        render_sec = render_duration(render_cpu)
        if i < warmup:
            warmup_times.append(elapsed)
        else:
            timing.append((elapsed, loaded - start, render_sec, max(0., elapsed - (loaded - start) - render_sec)))
        if i == 0 or i + 1 == warmup or (i + 1) % 10 == 0 or i + 1 == warmup + steps:
            print(f'train {i + 1}/{warmup + steps}, loss={loss_value:.5f}', flush=True)
    del it
    timing = np.asarray(timing)
    train_sec, wait_sec, render_sec, network_sec = timing.mean(axis=0).tolist()
    # Conservative: counts first-use GPU warmup as well as worker startup.
    train_startup = train_iterator_seconds + max(0., sum(warmup_times) - warmup * train_sec)
    net.eval()
    sync()
    start = time.perf_counter()
    vit = iter(vl)
    val_iterator_seconds = time.perf_counter() - start
    vtime, synth_times = [], []
    ps, ss = [], []
    with torch.no_grad():
        for i in range(val_steps):
            start = time.perf_counter()
            packed = next(vit)
            loaded = time.perf_counter()
            (x, y, _), render_cpu = timed_render(packed)
            pred = net(x)
            vp, vs = psnr(pred, y), ssim(pred, y)
            if not np.isfinite(vp).all() or not np.isfinite(vs).all():
                raise SystemExit('non-finite validation metrics')
            ps.append(vp)
            ss.append(vs)
            vtime.append(time.perf_counter() - start)
            synth_times.append(loaded - start + render_duration(render_cpu))
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == val_steps:
                print(f'validation {i + 1}/{val_steps}', flush=True)
    del vit
    val_psnr, val_ssim = float(np.concatenate(ps).mean()), float(np.concatenate(ss).mean())
    sync()
    epoch_measured = time.perf_counter() - epoch_start if a.full_epoch else None
    full_validation = val_steps == len(vl)
    if full_validation:
        val_epoch = val_iterator_seconds + sum(vtime)
    else:
        # Keep the observed prefix once; extrapolate only subsequent batches.
        val_tail = float(np.mean(vtime[1:] or vtime))
        val_epoch = val_iterator_seconds + sum(vtime) + (len(vl) - val_steps) * val_tail
    val_sec = val_epoch / len(vl)
    epoch = train_startup + train_sec * len(dl) + val_epoch
    if a.full_epoch:
        epoch = epoch_measured
    warnings = []
    if not a.full_epoch and (warmup < max(5, workers) or steps < max(32, 4 * workers)):
        warnings.append('Short training measurement: prefetch/startup can bias the estimate.')
    if not full_validation:
        warnings.append('Validation extrapolated; use --val-steps 0 for a complete pass.')
    half = max(1, len(timing) // 2)
    drift = float(timing[-half:, 0].mean() / timing[:half, 0].mean())
    if not .85 <= drift <= 1.15:
        warnings.append('Train speed changed by >15% between halves; do not use for final hardware sizing.')
    result = dict(
        benchmark_version=BENCHMARK_VERSION, physics_id=physics_id(), device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        level=a.level, dataset=counts, train_samples=len(ds),
        train_step_seconds=train_sec, train_data_wait_seconds=wait_sec,
        train_render_seconds=render_sec, train_network_seconds=network_sec,
        train_startup_estimate_seconds=train_startup,
        train_step_p90=float(np.quantile(timing[:, 0], .9)), train_second_half_ratio=drift,
        val_step_seconds=val_sec, val_epoch_estimate_seconds=val_epoch,
        val_first_step_seconds=vtime[0], validation_full_pass=full_validation,
        train_full_pass=a.full_epoch, epoch_measured_seconds=epoch_measured,
        train_l1=loss_sum / seen, val_psnr=val_psnr, val_ssim=val_ssim,
        optimizer_steps=warmup + steps, scheduler_total_steps=sched.T_max,
        lr_after_epoch=opt.param_groups[0]['lr'],
        epoch_estimate_seconds=epoch, hours_estimate=epoch * t['epochs'] / 3600,
        hours_with_20pct_reserve=epoch * t['epochs'] / 3600 * 1.2,
        epochs=t['epochs'], batch_size=batch, workers=workers,
        warmup_steps=warmup, measured_train_steps=steps, measured_val_steps=val_steps,
        train_batches=len(dl), val_batches=len(vl), test_batches=math.ceil(len(te) / batch),
        val_synthesis_seconds=float(np.mean(synth_times)),
        peak_cuda_GiB=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
        warnings=warnings,
        note='Estimate for this GPU and CPU allocation. Full-epoch mode measures wall time including '
             'worker shutdown and progress output. Partial mode estimates startup once per epoch. '
             'Checkpoint I/O and future contention or thermal drift are not measured. '
             'CUDA render time uses event markers, without per-stage synchronization; '
             'network time is the remaining step wall time, including optimizer and Python work. '
             'Data wait measures queue starvation, not total worker CPU cost.',
    )
    print(json.dumps(result, indent=2), flush=True)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
