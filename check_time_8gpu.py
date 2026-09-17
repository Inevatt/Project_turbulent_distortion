#!/usr/bin/env python3
"""SERVER ONLY: measure eight RTX 5090 concurrently; never start the campaign.

Place check_time_8gpu.py and train_8gpu.py together in the project root.
Activate the project's venv, then: python -u check_time_8gpu.py
Writes time_check_8gpu/summary.json and per-GPU benchmark logs.
Four original replicas, six levels, 60 epochs, batch=32, workers=14.
The benchmark uses disposable models; no campaign checkpoints are written.
No SSH, Git, package installation, data transfer, shutdown or training launch.
"""
import argparse

import csv

import datetime as dt

import hashlib

import json

import math

import os

from pathlib import Path, PurePosixPath

import re

import shlex

import shutil

import signal

import subprocess

import sys

import tempfile

import time

import traceback

# Limit BLAS before importing torch/numpy, including imports from train_8gpu.py.
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

VERSION = 2

PHYSICS = '170e863998bef7e311037c9b7e1116bc0be7e4f0b9995a98c35ddd2f1eb139e4'

PROTOCOL = '4ac9a67183576ea89c156eb024d39a6452c22a908170e0e003e94c6ae7d5af8c'

GIB = 1024 ** 3

LEVELS = ('d0', 'd1', 'd15', 'd2', 'd3', 'd3n')

STOP = False

def queues():
    return [dict(gpu=g, seed=g // 2 + 1,
                 levels=['d3', 'd15', 'd2'] if g % 2 == 0 else ['d3n', 'd0', 'd1'])
            for g in range(8)]

def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')

def say(message):
    print(f'[{stamp()}] {message}', flush=True)

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def execute(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)

def child_environment(project, gpu):
    env = os.environ.copy()
    # The IDs below refer to the devices visible to THIS container, not host IDs.
    visible = env.get('CUDA_VISIBLE_DEVICES')
    identifiers = visible.split(',') if visible else [str(i) for i in range(8)]
    if len(identifiers) < 8:
        raise RuntimeError('CUDA_VISIBLE_DEVICES exposes fewer than 8 GPUs')
    env['CUDA_VISIBLE_DEVICES'] = identifiers[gpu].strip()
    env['PYTHONPATH'] = str(project)
    env['PYTHONUNBUFFERED'] = '1'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = '1'
    return env

def terminate_children(children):
    for p in children:
        # A wrapper may already have exited while its training child is alive.
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    while any(p.poll() is None for p in children) and time.monotonic() < deadline:
        time.sleep(.1)
    for p in children:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()

def parallel_jobs(jobs, project, progress=None, timeout=None, lock=None):
    """Run independent process groups. Any failure stops siblings, preserving checkpoints."""
    active, streams = [], []
    started = time.monotonic()
    last_report = 0.
    try:
        for job in jobs:
            if STOP:
                raise InterruptedError('Stopped before launching the next worker')
            log = Path(job['log'])
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open('a', buffering=1)
            streams.append(stream)
            stream.write(f'\n[{stamp()}] COMMAND {shlex.join(job["command"])}\n')
            env = child_environment(project, job['gpu'])
            env.update(job.get('env', {}))
            p = subprocess.Popen(job['command'], cwd=project, env=env,
                                 stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True,
                                 pass_fds=(() if lock is None else (lock.fileno(),)))
            active.append((p, job))
        while True:
            if STOP:
                raise InterruptedError('Campaign stopped; completed epoch checkpoints are preserved')
            failed = [(p, j) for p, j in active if p.poll() not in (None, 0)]
            if failed:
                p, job = failed[0]
                tail = '\n'.join(Path(job['log']).read_text(errors='replace').splitlines()[-18:])
                raise RuntimeError(f'{job["name"]} exited {p.returncode}. {job["log"]}\n{tail}')
            running = [(p, j) for p, j in active if p.poll() is None]
            if not running:
                return
            now = time.monotonic()
            if timeout is not None and now - started > timeout:
                raise RuntimeError(f'Concurrent check exceeded {timeout / 60:g} minutes')
            if now - last_report >= 60:
                if progress:
                    progress([j['name'] for _, j in running])
                last_report = now
            time.sleep(.5)
    finally:
        terminate_children([p for p, _ in active])
        for stream in streams:
            stream.close()

def checkpoint_epoch(path, cfg, level, steps_per_epoch=None):
    import torch
    if not path.exists():
        return 0
    s = torch.load(path, map_location='cpu', weights_only=False)
    for key, expected in [('cfg', cfg), ('level', level), ('physics_id', PHYSICS),
                          ('protocol_id', PROTOCOL)]:
        if s.get(key) != expected:
            raise RuntimeError(f'{path}: incompatible {key}; existing results were not overwritten')
    epoch = s.get('epoch')
    if type(epoch) is not int or not 0 <= epoch <= cfg['train']['epochs']:
        raise RuntimeError(f'{path}: invalid epoch')
    for key in ('model', 'opt', 'sched'):
        if not isinstance(s.get(key), dict):
            raise RuntimeError(f'{path}: missing or invalid {key} state')
    # Validate actual tensors, not just an 'epoch=60' label.
    from src.unet import UNet
    m = cfg['model']
    with torch.random.fork_rng(devices=[]):
        net = UNet(in_ch=m['channels'], out_ch=m['channels'], base=m['base'], depth=m['depth'])
    net.load_state_dict(s['model'], strict=True)
    def finite_tree(value):
        if isinstance(value, torch.Tensor):
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                raise RuntimeError(f'{path}: non-finite checkpoint tensor')
        elif isinstance(value, dict):
            for v in value.values(): finite_tree(v)
        elif isinstance(value, (list, tuple)):
            for v in value: finite_tree(v)
    finite_tree(s['model'])
    finite_tree(s['opt'])
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg['train']['lr']))
    opt.load_state_dict(s['opt'])
    if epoch > 0:
        for parameter in net.parameters():
            state = opt.state.get(parameter, {})
            for key in ('exp_avg', 'exp_avg_sq'):
                if not isinstance(state.get(key), torch.Tensor) or state[key].shape != parameter.shape:
                    raise RuntimeError(f'{path}: invalid Adam {key} tensor')
    schedule = s['sched']
    total = schedule.get('T_max')
    last = schedule.get('last_epoch')
    if type(total) is not int or total <= 0 or type(last) is not int or last < 0:
        raise RuntimeError(f'{path}: invalid scheduler state')
    if steps_per_epoch is not None:
        if total != cfg['train']['epochs']*steps_per_epoch or last != epoch*steps_per_epoch:
            raise RuntimeError(f'{path}: scheduler budget/step differs from the dataset and epoch')
    if epoch > 0 and (not opt.state or any(int(v.get('step', -1)) != last for v in opt.state.values())):
        raise RuntimeError(f'{path}: optimizer step differs from scheduler')
    expected_lr = float(cfg['train']['lr'])*(1+math.cos(math.pi*last/total))/2
    if any(not math.isclose(float(g['lr']), expected_lr, rel_tol=1e-8, abs_tol=1e-12) for g in opt.param_groups):
        raise RuntimeError(f'{path}: learning rate differs from the configured cosine schedule')
    return epoch


def training_steps(project, configs):
    from src.data import split_indices
    result = {}
    for seed,cfg in configs.items():
        d = cfg['data']
        train,_,_ = split_indices(project/d['tiles'], d['val_frac'], d['test_frac'], cfg['split_seed'])
        result[seed] = len(train)*int(d['samples_per_tile'])//int(cfg['train']['batch_size'])
        if result[seed] < 1:
            raise RuntimeError(f'seed{seed}: empty training loader')
    return result


def check_level_locks(project, configs):
    """Also detect a surviving/manual trainer outside this controller."""
    import fcntl
    for cfg in configs.values():
        for level in LEVELS:
            path = project/cfg['out_dir']/level/'run.lock'
            if path.exists():
                with path.open('a') as handle:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise RuntimeError(f'A training process is still using {path.parent}')


def repair_resume_log(path, epoch):
    """Drop an interrupted tail, preserving the original log before a repair."""
    import io
    if not path.exists():
        return
    original = path.read_text(encoding='utf-8')
    reader = csv.DictReader(io.StringIO(original))
    fields = ['epoch', 'step', 'lr', 'train_l1', 'val_psnr', 'val_ssim', 'sec']
    if reader.fieldnames != fields:
        raise RuntimeError(f'{path}: invalid CSV header; log was preserved')
    kept = []
    for row in reader:
        try:
            index = int(row['epoch'])
            complete = set(row) == set(fields) and all(row[k] is not None for k in fields)
        except (ValueError, TypeError, KeyError):
            index, complete = -1, False
        if len(kept) == epoch:
            continue  # Uncommitted/truncated tail after the checkpoint.
        if not complete or index != len(kept)+1:
            raise RuntimeError(f'{path}: corruption before checkpoint epoch {epoch}; log was preserved')
        kept.append(row)
    # Missing logged epochs do not prevent restoring weights, but never fabricate them.
    if len(kept) != epoch:
        raise RuntimeError(f'{path}: log does not contain all {epoch} committed epochs; log was preserved')
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator='\n')
    writer.writeheader(); writer.writerows(kept)
    fixed = out.getvalue()
    if original != fixed:
        backup = path.with_name(path.name+f'.before_resume_{time.time_ns()}')
        backup.write_text(original, encoding='utf-8')
        tmp = path.with_suffix('.repair.tmp')
        tmp.write_text(fixed, encoding='utf-8')
        tmp.replace(path)
        say(f'Removed uncommitted CSV tail: {path}; original saved as {backup.name}')

def checked_configs(project):
    import yaml
    configs = {}
    for seed in range(1, 5):
        cfg = yaml.safe_load((project / f'configs/seed{seed}.yaml').read_text())
        expected = dict(epochs=60, batch_size=32, lr=.0003, num_workers=14)
        if cfg['train'] != expected or cfg['model'] != dict(channels=1, base=48, depth=3):
            raise RuntimeError(f'seed{seed}: training/model differs from the measured protocol')
        if (cfg['seed'], cfg['split_seed'], cfg['eval_seed']) != (seed * 1000, 41 + seed, 999):
            raise RuntimeError(f'seed{seed}: unexpected seeds')
        if cfg['out_dir'] != f'experiments_audit_v1_s{seed}':
            raise RuntimeError(f'seed{seed}: unexpected out_dir')
        if cfg['data'] != dict(root='data/clean', tiles='data/tiles', crop_px=128,
                               d_over_r0_range=[1., 5.], samples_per_tile=1,
                               val_frac=.05, test_frac=.1, margin_px=40):
            raise RuntimeError(f'seed{seed}: data configuration differs from the benchmark')
        if cfg['degradation'] != dict(diffraction_fwhm_px=2.):
            raise RuntimeError(f'seed{seed}: degradation configuration differs')
        configs[seed] = cfg
    return configs

def hardware_and_data(project):
    import numpy as np
    import torch
    from src.physics_version import physics_id, protocol_id
    if physics_id() != PHYSICS or protocol_id() != PROTOCOL:
        raise RuntimeError('Code differs from the checked physics/training protocol; no training started')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError('This launcher requires exactly 8 visible CUDA GPUs')
    names = [torch.cuda.get_device_name(i) for i in range(8)]
    if any('5090' not in name for name in names):
        raise RuntimeError(f'Expected eight RTX 5090, found: {names}')
    for i in range(8):
        x = torch.ones((128, 128), device=f'cuda:{i}', dtype=torch.complex64)
        torch.fft.fft2(x)
        torch.cuda.synchronize(i)
        del x
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    shm = shutil.disk_usage('/dev/shm')
    # Eight prepared D3 queues: (workers + in-flight) * batch * coeff/image bytes.
    needed_shm = math.ceil(8 * 16 * 32 * 4 * (37 * 208 ** 2 + 128 ** 2) * 1.15)
    if shm.free < needed_shm:
        raise RuntimeError(f'/dev/shm has {shm.free/GIB:.1f} GiB free; need '
                           f'{needed_shm/GIB:.1f}. Configure shared memory >=32 GiB in Vast.')
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            mem[parts[0].rstrip(':')] = int(parts[1]) * 1024
    limits = []
    for name in ('memory.max', 'memory/memory.limit_in_bytes'):
        f = Path('/sys/fs/cgroup') / name
        if f.exists():
            value = f.read_text().strip()
            if value.isdigit():
                used_file = f.with_name('memory.current' if name == 'memory.max' else 'memory.usage_in_bytes')
                used = int(used_file.read_text()) if used_file.exists() else 0
                limits.append(max(0, int(value) - used))
    effective_memory = min([mem.get('MemAvailable', 0), *limits])
    if effective_memory < 128 * GIB:
        raise RuntimeError(f'Only {effective_memory/GIB:.1f} GiB RAM available; need >=128 GiB')
    if shutil.disk_usage(project).free < 15 * GIB:
        raise RuntimeError('Need >=15 GiB free disk after environment and tiles installation')
    tiles = np.load(project / 'data/tiles/tiles.npy', mmap_mode='r')
    sources = np.load(project / 'data/tiles/source_id.npy', allow_pickle=False)
    if tiles.dtype != np.uint8 or tiles.ndim != 3 or tiles.shape[1:] != (256, 256):
        raise RuntimeError(f'Expected uint8 (N,256,256) tiles; got {tiles.shape}, {tiles.dtype}')
    if sources.ndim != 1 or len(tiles) != len(sources):
        raise RuntimeError('source_id.npy does not match tiles.npy')
    inputs = {name: digest(project/'data/tiles'/name) for name in ('tiles.npy','source_id.npy')}
    return dict(inputs=inputs, tiles=len(tiles), sources=len(np.unique(sources)), gpus=names, cpu_affinity=len(os.sched_getaffinity(0)),
                visibility={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','NVIDIA_VISIBLE_DEVICES')},
                runtime={k:__import__(k).__version__ for k in ('numpy','scipy','skimage','yaml')},
                cpu_max=(Path('/sys/fs/cgroup/cpu.max').read_text().strip()
                         if Path('/sys/fs/cgroup/cpu.max').exists() else None),
                ram_available_GiB=effective_memory/GIB, shm_free_GiB=shm.free/GIB,
                torch=torch.__version__, cuda=torch.version.cuda,
                driver=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version,power.limit',
                                                '--format=csv,noheader'], text=True).strip())

def make_forecast(measurements, remaining_epochs, include_eval=True):
    rows = []
    by_seed = {s: {} for s in range(1, 5)}
    for queue in queues():
        gpu, seed = queue['gpu'], queue['seed']
        hours = 0.
        for level in queue['levels']:
            x = measurements[(gpu, level)]
            if (x['physics_id'] != PHYSICS or x['benchmark_version'] < 4 or
                    not x['validation_full_pass'] or x['measured_train_steps'] < 64 or
                    x['batch_size'] != 32 or x['workers'] != 14 or x['epochs'] != 60 or
                    x['level'] != level):
                raise RuntimeError(f'Invalid benchmark for GPU {gpu}/{level}')
            if x['warnings']:
                raise RuntimeError(f'GPU {gpu}/{level}: {x["warnings"]}')
            seconds = float(x['epoch_estimate_seconds'])
            if not math.isfinite(seconds) or seconds <= 0:
                raise RuntimeError('Invalid benchmark duration')
            hours += seconds * remaining_epochs[(seed, level)] / 3600
            by_seed[seed][level] = x
        rows.append(dict(**queue, training_hours=hours, with_20pct_reserve=hours*1.2))
    eval_hours = []
    for seed, values in by_seed.items():
        # Planning heuristic only: synthesize once and count seven metric/forward blocks.
        # Actual evaluation uses six models plus no-op; we do not claim this is a hard bound.
        eval_hours.append(sum(5*x['test_batches']*(x['val_synthesis_seconds'] +
                          7*max(0., x['val_step_seconds']-x['val_synthesis_seconds']))
                          for x in values.values()) / 3600 * 1.2)
    training = max(row['with_20pct_reserve'] for row in rows)
    evaluation = max(eval_hours) if include_eval else 0.
    return dict(assignments=rows, training_hours_with_reserve=training,
                synthetic_eval_hours_heuristic=evaluation,
                planned_total_hours=training+evaluation,
                note='Estimate, not a time guarantee. Setup, transfer, benchmark, real-data '
                     'evaluation and real-data ceiling training are excluded.')


def project_root(value):
    project = Path(value).resolve()
    if not (project/'src/train.py').is_file() or not (project/'configs/seed1.yaml').is_file():
        raise RuntimeError('Run from the project root, or pass --project /path/to/project')
    os.chdir(project)
    sys.path.insert(0, str(project))
    return project


def acquire_lock(project):
    import fcntl
    path = project/'campaign_8gpu'
    path.mkdir(exist_ok=True)
    lock = (path/'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('A timing check or training campaign is already running in this project')
    return lock


def install_signals():
    global STOP
    STOP = False
    def handle(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def run_time_check(args):
    project = project_root(args.project)
    lock = acquire_lock(project)
    install_signals()
    root = project/'time_check_8gpu'
    root.mkdir(exist_ok=True)
    try:
        # Invalidate an earlier success until this attempt has completed.
        atomic_json(root/'summary.json', dict(status='running', started=stamp()))
        say(f'Timing launcher version {VERSION}')
        say('TIMING ONLY: checking hardware, data and the original seed1..seed4 configs')
        configs = checked_configs(project)
        check_level_locks(project, configs)
        hardware = hardware_and_data(project)
        run_dir = Path(tempfile.mkdtemp(prefix='run_', dir=root))
        bench_file = run_dir/'bench_training.py'
        bench_file.write_text(BENCHMARK_SOURCE)
        measurements = {}
        for stage in range(3):
            barrier = run_dir/f'barrier_{stage}'
            barrier.mkdir()
            jobs = []
            for queue in queues():
                gpu, seed, level = queue['gpu'], queue['seed'], queue['levels'][stage]
                output = run_dir/f'gpu{gpu}_{level}.json'
                jobs.append(dict(name=f'gpu{gpu}/{level}', gpu=gpu,
                    command=[sys.executable, '-u', str(bench_file), '--config', f'configs/seed{seed}.yaml',
                             '--level', level, '--device', 'cuda:0', '--warmup', '28', '--steps', '64',
                             '--val-steps', '0', '--output', str(output)],
                    log=output.with_suffix('.log'), env={
                        'TURBDIST_BENCH_BARRIER': str(barrier), 'TURBDIST_BENCH_SLOT': str(gpu)}))
            say(f'Concurrent timing round {stage+1}/3: 8 GPUs, 28 warmup + 64 timed steps + full validation')
            parallel_jobs(jobs, project, lambda names: say('Running: '+', '.join(names)), timeout=1200, lock=lock)
            for queue in queues():
                gpu, level = queue['gpu'], queue['levels'][stage]
                measurements[(gpu, level)] = json.loads((run_dir/f'gpu{gpu}_{level}.json').read_text())
        remaining = {(s,l):60 for s in range(1,5) for l in LEVELS}
        forecast = make_forecast(measurements, remaining)
        report = dict(version=VERSION, status='complete', completed=stamp(), physics_id=PHYSICS, protocol_id=PROTOCOL,
                      configs=configs, hardware=hardware, forecast=forecast,
                      measurements=[dict(slot=gpu, **x) for (gpu,level),x in measurements.items()])
        atomic_json(root/'summary.json', report)
        say('Timing complete. Forecast for a NEW campaign:')
        for q in forecast['assignments']:
            say(f'GPU {q["gpu"]}, seed{q["seed"]}, {" -> ".join(q["levels"])}: '
                f'{q["training_hours"]:.2f} h; with 20% reserve {q["with_20pct_reserve"]:.2f} h')
        say(f'TRAINING with 20% reserve: {forecast["training_hours_with_reserve"]:.2f} h')
        say(f'Optional synthetic evaluation: ~{forecast["synthetic_eval_hours_heuristic"]:.2f} h extra')
        if forecast['training_hours_with_reserve'] > 24:
            say('ATTENTION: forecast exceeds 24 hours. No campaign was started.')
        say('Report: '+str(root/'summary.json'))
        say('DONE. Training has NOT been started. Run train_8gpu.py separately when ready.')
        return 0
    except BaseException as e:
        atomic_json(root/'summary.json', dict(status='failed', updated=stamp(), error=str(e)))
        raise
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project', default='.', help='Project root; default current directory')
    return run_time_check(p.parse_args())

BENCHMARK_SOURCE = '"""Benchmark real training; measure loader startup once per estimated epoch.\n\nDefault: drain the prefetch queue, then time training and a full validation pass.\nNo checkpoint is written; model, physics and experiment config are unchanged.\n"""\nimport os\nfor key in (\'OMP_NUM_THREADS\', \'MKL_NUM_THREADS\', \'OPENBLAS_NUM_THREADS\'):\n    os.environ[key] = \'1\'\n\nimport argparse\nimport json\nimport math\nimport time\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\nimport yaml\nfrom torch.utils.data import DataLoader\nfrom src.data import split_indices\nfrom src.distortion import LEVELS\nfrom src.metrics import psnr, ssim\nfrom src.pair_pipeline import pairs_class, render_batch\nfrom src.physics_version import physics_id\nfrom src.unet import UNet\n\nBENCHMARK_VERSION = 6\n\n\ndef main():\n    p = argparse.ArgumentParser(description=__doc__)\n    p.add_argument(\'--config\', default=\'configs/seed1.yaml\')\n    p.add_argument(\'--level\', default=\'d3\', choices=sorted(LEVELS))\n    p.add_argument(\'--steps\', type=int, default=None,\n                   help=\'measured train batches; default max(64, 4 * workers)\')\n    p.add_argument(\'--warmup\', type=int, default=None,\n                   help=\'warmup train batches; default max(5, 2 * workers)\')\n    p.add_argument(\'--val-steps\', type=int, default=0,\n                   help=\'0 = complete validation pass (recommended)\')\n    p.add_argument(\'--full-epoch\', action=\'store_true\',\n                   help=\'time all train and validation batches; keep configured LR budget\')\n    p.add_argument(\'--device\', default=\'cuda\')\n    p.add_argument(\'--output\', type=Path)\n    a = p.parse_args()\n    if a.full_epoch and (a.steps is not None or a.warmup is not None or a.val_steps != 0):\n        p.error(\'--full-epoch cannot be combined with step-count overrides\')\n    cfg = yaml.safe_load(Path(a.config).read_text())\n    print(f\'benchmark_version={BENCHMARK_VERSION}\', flush=True)\n    d, t, m = cfg[\'data\'], cfg[\'train\'], cfg[\'model\']\n    workers, batch = int(t[\'num_workers\']), int(t[\'batch_size\'])\n    warmup = max(5, 2 * workers) if a.warmup is None else a.warmup\n    steps = max(64, 4 * workers) if a.steps is None else a.steps\n    if steps < 1 or warmup < 0 or a.val_steps < 0:\n        p.error(\'invalid step count\')\n    device = torch.device(a.device)\n    if device.type == \'cuda\' and not torch.cuda.is_available():\n        raise SystemExit(\'CUDA unavailable; run on the training server\')\n    torch.set_num_threads(1)\n    if device.type == \'cuda\':\n        if device.index is None:\n            device = torch.device(\'cuda\', torch.cuda.current_device())\n        torch.cuda.set_device(device)\n        torch.cuda.reset_peak_memory_stats(device)\n    torch.backends.cudnn.benchmark = False\n    torch.backends.cudnn.deterministic = True\n    deg = LEVELS[a.level](cfg[\'degradation\'][\'diffraction_fwhm_px\'])\n    tr, va, te = split_indices(d[\'tiles\'], d[\'val_frac\'], d[\'test_frac\'], cfg[\'split_seed\'])\n    kw = dict(tiles_dir=d[\'tiles\'], degradation=deg, margin_px=d[\'margin_px\'],\n              crop_px=d[\'crop_px\'], d_over_r0_range=d[\'d_over_r0_range\'])\n    ds = pairs_class(deg)(indices=tr, seed=cfg[\'seed\'],\n                          samples_per_tile=d[\'samples_per_tile\'], **kw)\n    vd = pairs_class(deg, True)(indices=va, seed=cfg[\'eval_seed\'], **kw)\n    if hasattr(deg, \'sampler\'):\n        deg.sampler._prepare(d[\'crop_px\'] + 2 * d[\'margin_px\'])\n    dlkw = dict(batch_size=batch, num_workers=workers, pin_memory=device.type == \'cuda\',\n                **({\'prefetch_factor\': 1} if workers else {}))\n    gen = torch.Generator().manual_seed(cfg[\'seed\'])\n    dl = DataLoader(ds, shuffle=True, generator=gen, drop_last=True, **dlkw)\n    vl = DataLoader(vd, shuffle=False, **dlkw)\n    if a.full_epoch:\n        warmup, steps = 0, len(dl)\n    val_steps = a.val_steps or len(vl)\n    if steps < 1 or len(dl) < steps + warmup or not val_steps or len(vl) < val_steps:\n        raise SystemExit(\'not enough batches for requested benchmark\')\n    source = np.load(Path(d[\'tiles\']) / \'source_id.npy\')\n    counts = {\n        name: dict(tiles=len(part), sources=int(len(np.unique(source[part]))))\n        for name, part in [(\'train\', tr), (\'val\', va), (\'test\', te)]\n    }\n    print(\'dataset: \' + json.dumps(counts), flush=True)\n    print(f\'train warmup={warmup}, measured={steps}; validation={val_steps}/{len(vl)}\', flush=True)\n    torch.manual_seed(cfg[\'seed\'])\n    net = UNet(in_ch=m[\'channels\'], out_ch=m[\'channels\'], base=m[\'base\'], depth=m[\'depth\']).to(device)\n    opt = torch.optim.Adam(net.parameters(), lr=float(t[\'lr\']))\n    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dl) * int(t[\'epochs\']))\n    loss_fn = torch.nn.L1Loss()\n    ds.set_epoch(0)\n    gen.manual_seed(cfg[\'seed\'])\n    net.train()\n\n    # Event markers measure the CUDA stream without inserting a barrier\n    # between synthesis and the network. Read them only after the ordinary\n    # loss.item()/metric CPU transfer has already completed that work.\n    render_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if device.type == \'cuda\' else None\n\n    def timed_render(packed):\n        start = time.perf_counter()\n        if render_events:\n            render_events[0].record()\n        result = render_batch(packed, deg, device)\n        if render_events:\n            render_events[1].record()\n        return result, time.perf_counter() - start\n\n    def render_duration(cpu_seconds):\n        if render_events:\n            return render_events[0].elapsed_time(render_events[1]) / 1000.\n        return cpu_seconds\n\n    def sync():\n        if device.type == \'cuda\':\n            torch.cuda.synchronize(device)\n\n    sync()\n    epoch_start = time.perf_counter()\n    start = time.perf_counter()\n    it = iter(dl)\n    train_iterator_seconds = time.perf_counter() - start\n    timing, warmup_times = [], []\n    loss_sum, seen = 0., 0\n    for i in range(warmup + steps):\n        if i == warmup and os.environ.get(\'TURBDIST_BENCH_BARRIER\'):\n            # Synchronize the start of measurement across eight independent GPUs.\n            # Waiting is outside the measured step and excluded from startup estimate.\n            sync()\n            barrier = Path(os.environ[\'TURBDIST_BENCH_BARRIER\'])\n            (barrier / (os.environ[\'TURBDIST_BENCH_SLOT\'] + \'.ready\')).touch()\n            deadline = time.monotonic() + 180\n            while len(list(barrier.glob(\'*.ready\'))) < 8:\n                if time.monotonic() > deadline:\n                    raise SystemExit(\'Timed out waiting for the other benchmark GPUs\')\n                time.sleep(.1)\n        start = time.perf_counter()\n        packed = next(it)\n        loaded = time.perf_counter()\n        (x, y, _), render_cpu = timed_render(packed)\n        loss = loss_fn(net(x), y)\n        if not torch.isfinite(loss):\n            raise SystemExit(\'non-finite training loss\')\n        opt.zero_grad(set_to_none=True)\n        loss.backward()\n        opt.step()\n        sched.step()\n        loss_value = loss.item()  # same synchronization/aggregation as train.py\n        loss_sum += loss_value * len(x)\n        seen += len(x)\n        elapsed = time.perf_counter() - start\n        render_sec = render_duration(render_cpu)\n        if i < warmup:\n            warmup_times.append(elapsed)\n        else:\n            timing.append((elapsed, loaded - start, render_sec, max(0., elapsed - (loaded - start) - render_sec)))\n        if i == 0 or i + 1 == warmup or (i + 1) % 10 == 0 or i + 1 == warmup + steps:\n            print(f\'train {i + 1}/{warmup + steps}, loss={loss_value:.5f}\', flush=True)\n    shutdown_start = time.perf_counter()\n    del it\n    train_shutdown_seconds = time.perf_counter() - shutdown_start\n    timing = np.asarray(timing)\n    train_sec, wait_sec, render_sec, network_sec = timing.mean(axis=0).tolist()\n    # Conservative: counts first-use GPU warmup as well as worker startup.\n    train_startup = train_iterator_seconds + max(0., sum(warmup_times) - warmup * train_sec)\n    net.eval()\n    sync()\n    start = time.perf_counter()\n    vit = iter(vl)\n    val_iterator_seconds = time.perf_counter() - start\n    vtime, synth_times = [], []\n    ps, ss = [], []\n    with torch.no_grad():\n        for i in range(val_steps):\n            start = time.perf_counter()\n            packed = next(vit)\n            loaded = time.perf_counter()\n            (x, y, _), render_cpu = timed_render(packed)\n            pred = net(x)\n            vp, vs = psnr(pred, y), ssim(pred, y)\n            if not np.isfinite(vp).all() or not np.isfinite(vs).all():\n                raise SystemExit(\'non-finite validation metrics\')\n            ps.append(vp)\n            ss.append(vs)\n            vtime.append(time.perf_counter() - start)\n            synth_times.append(loaded - start + render_duration(render_cpu))\n            if i == 0 or (i + 1) % 10 == 0 or i + 1 == val_steps:\n                print(f\'validation {i + 1}/{val_steps}\', flush=True)\n    shutdown_start = time.perf_counter()\n    del vit\n    val_shutdown_seconds = time.perf_counter() - shutdown_start\n    val_psnr, val_ssim = float(np.concatenate(ps).mean()), float(np.concatenate(ss).mean())\n    sync()\n    epoch_measured = time.perf_counter() - epoch_start if a.full_epoch else None\n    full_validation = val_steps == len(vl)\n    if full_validation:\n        val_epoch = val_iterator_seconds + sum(vtime) + val_shutdown_seconds\n    else:\n        # Keep the observed prefix once; extrapolate only subsequent batches.\n        val_tail = float(np.mean(vtime[1:] or vtime))\n        val_epoch = val_iterator_seconds + sum(vtime) + val_shutdown_seconds + (len(vl) - val_steps) * val_tail\n    val_sec = val_epoch / len(vl)\n    epoch = train_startup + train_sec * len(dl) + train_shutdown_seconds + val_epoch\n    if a.full_epoch:\n        epoch = epoch_measured\n    warnings = []\n    if not a.full_epoch and (warmup < max(5, workers) or steps < max(32, 4 * workers)):\n        warnings.append(\'Short training measurement: prefetch/startup can bias the estimate.\')\n    if not full_validation:\n        warnings.append(\'Validation extrapolated; use --val-steps 0 for a complete pass.\')\n    half = max(1, len(timing) // 2)\n    drift = float(timing[-half:, 0].mean() / timing[:half, 0].mean())\n    if not .85 <= drift <= 1.15:\n        warnings.append(\'Train speed changed by >15% between halves; do not use for final hardware sizing.\')\n    result = dict(\n        benchmark_version=BENCHMARK_VERSION, physics_id=physics_id(), device=str(device),\n        gpu=torch.cuda.get_device_name(device) if device.type == \'cuda\' else None,\n        level=a.level, dataset=counts, train_samples=len(ds),\n        train_step_seconds=train_sec, train_data_wait_seconds=wait_sec,\n        train_render_seconds=render_sec, train_network_seconds=network_sec,\n        train_startup_estimate_seconds=train_startup,\n        train_shutdown_seconds=train_shutdown_seconds,\n        val_shutdown_seconds=val_shutdown_seconds,\n        train_step_p90=float(np.quantile(timing[:, 0], .9)), train_second_half_ratio=drift,\n        val_step_seconds=val_sec, val_epoch_estimate_seconds=val_epoch,\n        val_first_step_seconds=vtime[0], validation_full_pass=full_validation,\n        train_full_pass=a.full_epoch, epoch_measured_seconds=epoch_measured,\n        train_l1=loss_sum / seen, val_psnr=val_psnr, val_ssim=val_ssim,\n        optimizer_steps=warmup + steps, scheduler_total_steps=sched.T_max,\n        lr_after_epoch=opt.param_groups[0][\'lr\'],\n        epoch_estimate_seconds=epoch, hours_estimate=epoch * t[\'epochs\'] / 3600,\n        hours_with_20pct_reserve=epoch * t[\'epochs\'] / 3600 * 1.2,\n        epochs=t[\'epochs\'], batch_size=batch, workers=workers,\n        warmup_steps=warmup, measured_train_steps=steps, measured_val_steps=val_steps,\n        train_batches=len(dl), val_batches=len(vl), test_batches=math.ceil(len(te) / batch),\n        val_synthesis_seconds=float(np.mean(synth_times)),\n        peak_cuda_GiB=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == \'cuda\' else None,\n        warnings=warnings,\n        note=\'Estimate for this GPU and CPU allocation. Full-epoch mode measures wall time including \'\n             \'worker shutdown and progress output. Partial mode estimates startup once per epoch. \'\n             \'Checkpoint I/O and future contention or thermal drift are not measured. \'\n             \'CUDA render time uses event markers, without per-stage synchronization; \'\n             \'network time is the remaining step wall time, including optimizer and Python work. \'\n             \'Data wait measures queue starvation, not total worker CPU cost.\',\n    )\n    print(json.dumps(result, indent=2), flush=True)\n    if a.output:\n        a.output.parent.mkdir(parents=True, exist_ok=True)\n        a.output.write_text(json.dumps(result, indent=2) + \'\\n\')\n\n\nif __name__ == \'__main__\':\n    main()\n'

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (Exception, KeyboardInterrupt) as exc:
        say(f'ERROR: {exc}')
        sys.exit(1)
