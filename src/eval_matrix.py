"""Synthetic cross-generalization matrix with shared degradation batches.

Generate each (column, D/r0, tile) once, then apply every restoration model.
Checkpoint each completed column/strength atomically. Final per-row NPZ files
retain psnr_{column}_dr{v}, ssim_{column}_dr{v}, tile_idx keys.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from .data import split_indices
from .physics_version import physics_id, protocol_id
from .pair_pipeline import pairs_class, render_batch
from .distortion import LEVELS
from .metrics import psnr, ssim
from .unet import UNet

TEST_LEVELS = tuple(LEVELS)
DR_GRID = (1., 2., 3., 4., 5.)
OUT = Path('results')


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_npz(path, **data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **data)
    tmp.replace(path)


def load_model(cfg, row, device):
    ckpt = Path(cfg['out_dir']) / row / 'last.pt'
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    for key, expected in [('cfg', cfg), ('level', row),
                          ('physics_id', physics_id()), ('protocol_id', protocol_id()),
                          ('epoch', int(cfg['train']['epochs']))]:
        if state.get(key) != expected:
            raise ValueError(f'{ckpt}: incompatible or incomplete checkpoint ({key})')
    m = cfg['model']
    net = UNet(in_ch=m['channels'], out_ch=m['channels'], base=m['base'], depth=m['depth']).to(device)
    net.load_state_dict(state['model'])
    return net.eval()


def run_models(models, ds, batch, workers, device):
    """All rows receive the identical tensor, including on CUDA."""
    if hasattr(ds.deg, 'sampler'):
        ds.deg.sampler._prepare(ds.crop_px + 2 * ds.margin_px)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    pin_memory=device.type == 'cuda',
                    **({'prefetch_factor': 1} if workers else {}))
    scores = {row: ([], []) for row in models}
    with torch.no_grad():
        for packed in dl:
            degraded, clean, _ = render_batch(packed, ds.deg, device)
            for row, net in models.items():
                pred = degraded if net is None else net(degraded)
                scores[row][0].append(psnr(pred, clean))
                scores[row][1].append(ssim(pred, clean))
    return {row: (np.concatenate(p), np.concatenate(s)) for row, (p, s) in scores.items()}


def run_column(net, ds, batch, workers, device):
    return run_models({'row': net}, ds, batch, workers, device)['row']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', type=Path, default=OUT)
    ap.add_argument('--columns', nargs='+', choices=TEST_LEVELS, default=list(TEST_LEVELS))
    ap.add_argument('--dr-grid', nargs='+', type=float, default=list(DR_GRID))
    args = ap.parse_args()
    if not all(np.isfinite(v) and v >= 1 for v in args.dr_grid):
        ap.error('D/r0 must be finite and >= 1')
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    d, t = cfg['data'], cfg['train']
    tag = Path(cfg['out_dir']).name.replace('experiments_', '')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _, _, te = split_indices(d['tiles'], d['val_frac'], d['test_frac'], cfg['split_seed'])
    if not len(te):
        raise SystemExit('empty test split')
    # Validate ALL checkpoints before spending time on synthesis/no-op.
    models = {'noop': None}
    for row in LEVELS:
        models[row] = load_model(cfg, row, device)
    meta = dict(cfg=cfg, physics_id=physics_id(), protocol_id=protocol_id(),
                evaluator=file_sha256(__file__), columns=args.columns, dr_grid=args.dr_grid,
                checkpoints={row: file_sha256(Path(cfg['out_dir']) / row / 'last.pt') for row in LEVELS},
                dataset={name: file_sha256(Path(d['tiles']) / name) for name in ('source_id.npy', 'tiles.npy')})
    provenance = json.dumps(meta, sort_keys=True)
    kw = dict(crop_px=d['crop_px'], margin_px=d['margin_px'], seed=cfg['eval_seed'])
    all_scores = {row: {} for row in models}
    print(f'{tag}: {len(te)} test tiles, {len(models)} rows; synthesis shared across rows', flush=True)
    for col in args.columns:
        deg = LEVELS[col](cfg['degradation']['diffraction_fwhm_px'])
        for v in args.dr_grid:
            suffix = f'{col}_dr{v:g}'
            shard = args.output / '_parts' / tag / f'{suffix}.npz'
            if shard.exists():
                with np.load(shard, allow_pickle=False) as old:
                    if str(old['provenance']) != provenance or not np.array_equal(old['tile_idx'], te):
                        raise SystemExit(f'{shard}: provenance differs; choose a fresh --output')
                    result = {row: (old[f'{row}_psnr'].copy(), old[f'{row}_ssim'].copy()) for row in models}
            else:
                ds = pairs_class(deg, frozen=True)(d['tiles'], te, deg, d_over_r0_range=(v, v), **kw)
                result = run_models(models, ds, int(t['batch_size']), int(t['num_workers']), device)
                packed = {f'{row}_{metric}': arr.astype(np.float32)
                          for row, vals in result.items() for metric, arr in zip(('psnr', 'ssim'), vals)}
                atomic_npz(shard, provenance=provenance, tile_idx=te, **packed)
            for row, (p, s) in result.items():
                all_scores[row][f'psnr_{suffix}'] = p.astype(np.float32)
                all_scores[row][f'ssim_{suffix}'] = s.astype(np.float32)
            print(f'  {suffix}: ' + ' | '.join(f'{row} {p.mean():.2f}' for row, (p, _) in result.items()), flush=True)
    for row, data in all_scores.items():
        atomic_npz(args.output / f'{tag}__{row}.npz', physics_id=physics_id(),
                   protocol_id=protocol_id(), provenance=provenance, tile_idx=te, **data)
    print(f'done: {args.output}/{tag}__*.npz', flush=True)


if __name__ == '__main__':
    main()
