"""Short trainability diagnostic using the configured network and cached pairs.

Render each selected pair once. Repeating cached pairs is a diagnostic, NOT a
replacement for online training and NOT evidence for an epoch/convergence budget.
No experiment checkpoint or config is modified. The test split is never used.
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
from torch.utils.data import DataLoader, Subset

from src.data import split_indices
from src.distortion import LEVELS
from src.metrics import psnr, ssim
from src.pair_pipeline import pairs_class, render_batch
from src.physics_version import physics_id, protocol_id
from src.unet import UNet


def select_positions(indices, source_id, count, seed, repeats=1):
    """One random tile per selected source; preserve the full-dataset RNG index."""
    ids = source_id[indices]
    sources = np.unique(ids)
    if count > len(sources):
        raise ValueError(f'requested {count} sources, only {len(sources)} available')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(sources, count, replace=False)
    positions = [int(rng.choice(np.flatnonzero(ids == s)))
                 + int(rng.integers(repeats)) * len(indices) for s in chosen]
    return np.asarray(positions), chosen


def make_cache(ds, positions, deg, device, batch, workers, label):
    loader = DataLoader(Subset(ds, positions.tolist()), batch_size=batch,
                        num_workers=workers, pin_memory=device.type == 'cuda',
                        **({'prefetch_factor': 1} if workers else {}))
    xs, ys, strengths = [], [], []
    start = time.perf_counter()
    for i, packed in enumerate(loader):
        x, y, dr = render_batch(packed, deg, device)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('non-finite cached pair')
        xs.append(x.cpu())
        ys.append(y.cpu())
        strengths.append(dr.cpu())
        print(f'cache {label}: {i + 1}/{len(loader)} batches', flush=True)
    print(f'cache {label}: {time.perf_counter() - start:.1f} s', flush=True)
    return tuple(torch.cat(parts).to(device) for parts in (xs, ys, strengths))


@torch.no_grad()
def evaluate(net, cache, batch):
    if net is not None:
        net.eval()
    x, y, dr = cache
    values = {k: [] for k in ('l1_raw', 'l1_clipped', 'mse_clipped', 'psnr', 'ssim', 'correction_l1')}
    for start in range(0, len(x), batch):
        xx, yy = x[start:start + batch], y[start:start + batch]
        pred = xx if net is None else net(xx)
        if not torch.isfinite(pred).all():
            raise ValueError('non-finite network prediction')
        clipped = pred.clamp(0, 1)
        for key, tensor in (
            ('l1_raw', (pred - yy).abs()), ('l1_clipped', (clipped - yy).abs()),
            ('mse_clipped', (clipped - yy).square()), ('correction_l1', (pred - xx).abs()),
        ):
            values[key].append(tensor.flatten(1).mean(1).cpu().numpy())
        values['psnr'].append(psnr(pred, yy))
        values['ssim'].append(ssim(pred, yy))
    arrays = {key: np.concatenate(parts) for key, parts in values.items()}
    strengths = dr.cpu().numpy()
    result = {}
    groups = [('all', np.ones(len(x), dtype=bool)),
              ('D/r0<2', strengths < 2), ('2<=D/r0<4', (strengths >= 2) & (strengths < 4)),
              ('D/r0>=4', strengths >= 4)]
    for label, mask in groups:
        if mask.any():
            result[label] = dict(n=int(mask.sum()), **{key: float(v[mask].mean()) for key, v in arrays.items()})
    return result


def fit_phase(name, train_cache, val_cache, cfg, steps, total_steps, output, baseline):
    device = train_cache[0].device
    m, t = cfg['model'], cfg['train']
    torch.manual_seed(cfg['seed'])
    net = UNet(in_ch=m['channels'], out_ch=m['channels'], base=m['base'], depth=m['depth']).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(t['lr']))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    loss_fn = torch.nn.L1Loss()
    batch = min(int(t['batch_size']), len(train_cache[0]))
    x, y, _ = train_cache
    with torch.no_grad():
        identity_error = float((net(x[:batch]) - x[:batch]).abs().max())
    initial_trunk = net.inc.body[0][0].weight.detach().clone()
    checkpoints = sorted({s for s in (0, 10, 50, 100, 200, steps) if s <= steps})
    rows, gradients = [], []
    generator = torch.Generator().manual_seed(cfg['seed'])
    order, offset = None, len(x)
    start = time.perf_counter()

    def report(step):
        tr = evaluate(net, train_cache, batch)
        va = evaluate(net, val_cache, int(t['batch_size'])) if val_cache is not None else None
        row = dict(phase=name, step=step, train=tr, val=va,
                   elapsed_seconds=time.perf_counter() - start, lr=opt.param_groups[0]['lr'])
        rows.append(row)
        with (output / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        msg = (f'{name} step {step}: train L1={tr["all"]["l1_raw"]:.5f} '
               f'(no-op {baseline["train"]["all"]["l1_raw"]:.5f}), '
               f'train PSNR={tr["all"]["psnr"]:.2f}')
        if va:
            msg += (f'; val PSNR={va["all"]["psnr"]:.2f} '
                    f'(no-op {baseline["val"]["all"]["psnr"]:.2f}), '
                    f'val L1={va["all"]["l1_raw"]:.5f}')
        print(msg, flush=True)

    report(0)
    for step in range(1, steps + 1):
        if offset + batch > len(x):
            order = torch.randperm(len(x), generator=generator).to(device)
            offset = 0
        idx = order[offset:offset + batch]
        offset += batch
        net.train()
        pred = net(x[idx])
        loss = loss_fn(pred, y[idx])
        if not torch.isfinite(loss):
            raise ValueError('non-finite loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step in (1, 2, steps):
            gradients.append(dict(step=step,
                head_grad_mean_abs=float(net.head.weight.grad.abs().mean()),
                first_conv_grad_mean_abs=float(net.inc.body[0][0].weight.grad.abs().mean())))
        opt.step()
        sched.step()
        if step in checkpoints:
            report(step)
        elif step % 50 == 0:
            print(f'{name}: {step}/{steps} optimizer steps', flush=True)
    trunk_change = float((net.inc.body[0][0].weight.detach() - initial_trunk).abs().max())
    return dict(steps=steps, batch_size=batch, cached_train_pairs=len(x),
                identity_error_at_start=identity_error, gradients=gradients,
                first_conv_max_weight_change=trunk_change, scheduler_total_steps=total_steps,
                baseline=baseline, history=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/seed1.yaml')
    p.add_argument('--level', choices=sorted(LEVELS), default='d3')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--train-samples', type=int, default=256)
    p.add_argument('--val-samples', type=int, default=128)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--overfit-samples', type=int, default=8)
    p.add_argument('--overfit-steps', type=int, default=200)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--output', type=Path, default=Path('diagnostics/learning_d3'))
    a = p.parse_args()
    if min(a.train_samples, a.val_samples, a.steps, a.overfit_samples, a.overfit_steps) < 1:
        p.error('counts and steps must be positive')
    if a.overfit_samples > a.train_samples:
        p.error('overfit-samples exceeds train-samples')
    cfg = yaml.safe_load(Path(a.config).read_text())
    d, t = cfg['data'], cfg['train']
    device = torch.device(a.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            p.error('CUDA unavailable')
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_device(index)
        device = torch.device('cuda', index)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    workers = int(t['num_workers']) if a.workers is None else a.workers
    if workers < 0:
        p.error('workers must be nonnegative')
    tr, va, _ = split_indices(d['tiles'], d['val_frac'], d['test_frac'], cfg['split_seed'])
    sid = np.load(Path(d['tiles']) / 'source_id.npy')
    train_pos, train_sources = select_positions(tr, sid, a.train_samples, cfg['seed'] + 915, int(d['samples_per_tile']))
    val_pos, val_sources = select_positions(va, sid, a.val_samples, cfg['eval_seed'] + 915)
    if np.intersect1d(train_sources, val_sources).size:
        raise ValueError('train/validation source overlap')
    deg = LEVELS[a.level](cfg['degradation']['diffraction_fwhm_px'])
    kw = dict(tiles_dir=d['tiles'], degradation=deg, margin_px=d['margin_px'],
              crop_px=d['crop_px'], d_over_r0_range=d['d_over_r0_range'])
    train_ds = pairs_class(deg)(indices=tr, seed=cfg['seed'], samples_per_tile=d['samples_per_tile'], **kw)
    val_ds = pairs_class(deg, True)(indices=va, seed=cfg['eval_seed'], **kw)
    train_ds.set_epoch(0)
    if hasattr(deg, 'sampler'):
        deg.sampler._prepare(d['crop_px'] + 2 * d['margin_px'])
    total_steps = (len(train_ds) // int(t['batch_size'])) * int(t['epochs'])
    if max(a.steps, a.overfit_steps) > total_steps:
        p.error('diagnostic steps exceed configured training schedule')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'progress.jsonl').write_text('')
    print(f'DIAGNOSTIC ONLY: {a.train_samples} train sources + {a.val_samples} val sources; '
          f'original UNet/crop/L1/Adam/LR schedule. Each pair rendered once.', flush=True)
    start = time.perf_counter()
    train_cache = make_cache(train_ds, train_pos, deg, device, int(t['batch_size']), workers, 'train')
    val_cache = make_cache(val_ds, val_pos, deg, device, int(t['batch_size']), workers, 'val')
    baseline = dict(train=evaluate(None, train_cache, int(t['batch_size'])),
                    val=evaluate(None, val_cache, int(t['batch_size'])))
    small = tuple(x[:a.overfit_samples] for x in train_cache)
    overfit_baseline = dict(train=evaluate(None, small, min(a.overfit_samples, int(t['batch_size']))))
    result = dict(diagnostic_version=1, physics_id=physics_id(), protocol_id=protocol_id(),
                  config=cfg, device=str(device),
                  gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  level=a.level, train_positions=train_pos.tolist(), val_positions=val_pos.tolist(),
                  train_sources=train_sources.tolist(), val_sources=val_sources.tolist(),
                  warning='Cached fixed pairs: diagnostic only; cannot determine production convergence or epoch count. '
                          'Strength bins contain different images and are descriptive, not a paired severity experiment.')
    result['overfit'] = fit_phase('overfit', small, None, cfg, a.overfit_steps, total_steps, a.output, overfit_baseline)
    result['cached_subset'] = fit_phase('cached_subset', train_cache, val_cache, cfg, a.steps, total_steps, a.output, baseline)
    result['elapsed_seconds'] = time.perf_counter() - start
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'FINISHED in {result["elapsed_seconds"]:.1f} s; results: {a.output / "summary.json"}', flush=True)
    for key in ('overfit', 'cached_subset'):
        phase = result[key]
        final = phase['history'][-1]
        print(json.dumps(dict(phase=key, identity_error=phase['identity_error_at_start'],
              gradients=phase['gradients'], first_conv_max_weight_change=phase['first_conv_max_weight_change'],
              train_noop=phase['baseline']['train']['all'], train_final=final['train']['all'],
              val_noop=phase['baseline'].get('val', {}).get('all'),
              val_final=final['val']['all'] if final['val'] else None), indent=2), flush=True)


if __name__ == '__main__':
    main()
