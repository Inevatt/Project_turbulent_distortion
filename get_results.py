#!/usr/bin/env python3
"""Final evaluation only: four replicas, synthetic matrix, TSRWGAN and masked OTIS.

Place beside check_time_8gpu.py in the project root. Run after training:
  python -u get_results.py --check-only
  nohup python -u get_results.py > evaluation.log 2>&1 < /dev/null &
Restart the same command to reuse completed evaluation parts.
No training, checkpoint selection, image filtering or real-data fitting.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from collections import defaultdict
import csv
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

VERSION = 1
LEVELS = ('d0', 'd1', 'd15', 'd2', 'd3', 'd3n')
ROWS = ('noop', *LEVELS)
STRENGTHS = (1, 2, 3, 4, 5)
CORE_COLUMNS = ('d0', 'd1', 'd2', 'd3')
REAL = {
    'otis': ('data/real/otis/manifest_masked.json', 4328, 16),
    'tsrwgan': ('data/real/tsrwgan/screened_v2/manifest.json', 2230, 455),
}
CHUNK = 32
DENOM_EPS_DB = 1e-6  # numerical-zero guard, not a scientific effect-size threshold


def say(message):
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    print(f'[{stamp}] {message}', flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def json_write(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def csv_write(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode('utf-8'))


def read_manifest(path, dataset, strict=True):
    value = json.loads(Path(path).read_text())
    entries = value['samples'] if isinstance(value, dict) else value
    if not isinstance(entries, list) or not entries:
        raise ValueError(f'{path}: expected a nonempty sample list')
    ids = [e['id'] for e in entries]
    if len(set(ids)) != len(ids):
        raise ValueError(f'{path}: duplicate sample IDs')
    for e in entries:
        for key in ('id', 'scene', 'input', 'target'):
            if not isinstance(e[key], str) or not e[key]:
                raise ValueError(f'{path}: invalid {key}')
        if dataset == 'otis' and not e.get('mask'):
            raise ValueError('Every OTIS sample must have a mask')
        if dataset == 'tsrwgan' and e.get('mask'):
            raise ValueError('Unexpected mask in the frozen TSRWGAN protocol')
    if strict:
        _, count, scenes = REAL[dataset]
        if len(entries) != count or len({e['scene'] for e in entries}) != scenes:
            raise ValueError(f'{path}: expected {count} samples / {scenes} groups')
    return entries


def verify_sources(project, dataset):
    relative, count, scenes = REAL[dataset]
    manifest = project / relative
    entries = read_manifest(manifest, dataset)
    sums = manifest.parent / 'SHA256SUMS'
    expected = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split('  ', 1)
        expected[name] = digest
    needed = {manifest.name}
    for e in entries:
        needed.update(e[k] for k in ('input', 'target', 'mask') if k in e)
    fingerprint = hashlib.sha256()
    for name in sorted(needed):
        path = manifest.parent / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(manifest.parent.resolve()):
            raise ValueError(f'Unexpected dataset path: {name}')
        digest = sha(path)
        if expected.get(name) != digest:
            raise ValueError(f'{path}: missing or mismatched original SHA256')
        fingerprint.update(canonical([name, digest]).encode())
    say(f'{dataset}: {count} pairs / {scenes} groups, source checksums OK')
    return dict(manifest=relative, manifest_sha256=sha(manifest),
                source_fingerprint=fingerprint.hexdigest(), samples=count, groups=scenes)


def load_gray(path):
    with Image.open(path) as im:
        if im.mode != 'L':
            raise ValueError(f'{path}: expected the prepared 8-bit grayscale PNG')
        return np.asarray(im, dtype=np.float32) / 255.0


def metric_scores(pred, target, mask=None):
    """Same float64/clipping/SSIM convention as src.metrics, with mask support."""
    from scipy.ndimage import binary_erosion
    from skimage.metrics import structural_similarity
    p, t = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    if p.ndim != 2 or p.shape != t.shape or min(p.shape) < 11:
        raise ValueError('Invalid metric image shape')
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Non-finite prediction/target')
    p = np.clip(p, 0, 1)
    valid = np.ones(p.shape, bool) if mask is None else np.asarray(mask, bool)
    if valid.shape != p.shape or not valid.any():
        raise ValueError('Invalid/empty metric mask')
    mse = float(np.mean((p[valid] - t[valid]) ** 2))
    psnr = float(-10 * np.log10(max(mse, 1e-12)))
    kwargs = dict(data_range=1.0, gaussian_weights=True, sigma=1.5,
                  use_sample_covariance=False, win_size=11)
    if mask is None:
        ssim = float(structural_similarity(p, t, **kwargs))
    else:
        centres = binary_erosion(valid, structure=np.ones((11, 11), bool), border_value=0)
        if not centres.any():
            raise ValueError('Mask has no complete 11x11 SSIM windows')
        _, field = structural_similarity(p, t, full=True, **kwargs)
        ssim = float(field[centres].mean())
    return psnr, ssim


def metric_selftest():
    from src.metrics import psnr, ssim
    rng = np.random.default_rng(173)
    t = rng.random((41, 47), dtype=np.float32)
    p = t + rng.normal(0, .07, t.shape).astype(np.float32)
    ref = (float(psnr(p[None, None], t[None, None])[0]),
           float(ssim(p[None, None], t[None, None])[0]))
    np.testing.assert_allclose(metric_scores(p, t), ref, atol=1e-12, rtol=0)
    np.testing.assert_allclose(metric_scores(p, t, np.ones(t.shape, bool)), ref, atol=1e-12, rtol=0)
    mask = np.ones(t.shape, bool)
    mask[:6] = False
    mask[18:23, 18:23] = False
    before = metric_scores(p, t, mask)
    altered = p.copy()
    altered[~mask] = 1.0 - t[~mask]
    np.testing.assert_allclose(metric_scores(altered, t, mask), before, atol=1e-12, rtol=0)
    np.testing.assert_allclose(metric_scores(t, t, mask), (120., 1.), atol=1e-12, rtol=0)
    try:
        metric_scores(t, t, np.zeros(t.shape, bool))
    except ValueError:
        pass
    else:
        raise AssertionError('Empty mask was not rejected')
    say('Metric self-test: original conventions and masked-window invariants OK')


def preview(path, sample_id, images):
    # Visualisation only. Metrics above always use the native full resolution.
    names = ['input', 'target', *LEVELS]
    width = 480
    height = round(images['input'].shape[0] * width / images['input'].shape[1])
    canvas = Image.new('RGB', (width * 2, (height + 24) * 4 + 32), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    draw.text((5, 5), sample_id + ' | preview only; metrics at native resolution', fill='black', font=font)
    for i, name in enumerate(names):
        x, y = (i % 2) * width, 32 + (i // 2) * (height + 24)
        pixels = np.round(np.clip(images[name], 0, 1) * 255).astype(np.uint8)
        method = Image.Resampling.NEAREST if width > pixels.shape[1] else Image.Resampling.LANCZOS
        im = Image.fromarray(pixels).resize((width, height), method)
        draw.text((x + 4, y + 3), name, fill='black', font=font)
        canvas.paste(im, (x, y + 24))
    stream = io.BytesIO()
    canvas.save(stream, format='PNG')
    atomic_bytes(path, stream.getvalue())


def read_part(path, provenance, ids):
    with np.load(path, allow_pickle=False) as data:
        if str(data['provenance']) != provenance or list(data['sample_id']) != list(ids):
            raise ValueError(f'{path}: incompatible saved evaluation part')
        arrays = {k: data[k].copy() for k in data.files}
    for row in ROWS:
        for metric in ('psnr', 'ssim'):
            arr = arrays[f'{metric}_{row}']
            if arr.shape != (len(ids),) or not np.isfinite(arr).all():
                raise ValueError(f'{path}: invalid {metric}/{row}')
    return arrays


def real_worker(project, output, seed, device_name='cuda:0', strict=True):
    import torch
    from src.eval_matrix import atomic_npz, load_model
    from src.eval_real import predict_patches
    torch.set_num_threads(1)
    if device_name.startswith('cuda'):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expected exactly one CUDA GPU; refusing CPU fallback')
        torch.cuda.set_device(0)
    device = torch.device(device_name)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True  # matches the project's default inference
    plan = json.loads((output / 'evaluation_plan.json').read_text())
    config_path = project / f'configs/seed{seed}.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    if cfg != plan['configs'][str(seed)]:
        raise ValueError('Config changed after evaluation preflight')
    if sha(__file__) != plan['driver_sha256']:
        raise ValueError('Evaluation driver changed after preflight')
    models = {lv: load_model(cfg, lv, device) for lv in LEVELS}
    for lv in LEVELS:
        if sha(project / cfg['out_dir'] / lv / 'last.pt') != plan['checkpoints'][str(seed)][lv]:
            raise ValueError('Checkpoint changed after preflight')
    for dataset, (relative, _, _) in REAL.items():
        manifest = project / relative
        entries = read_manifest(manifest, dataset, strict=strict)
        provenance = canonical(dict(plan=plan, seed=seed, dataset=dataset))
        destination = output / 'real' / f'seed{seed}__{dataset}.npz'
        if destination.exists():
            read_part(destination, provenance, [e['id'] for e in entries])
            say(f'seed{seed}/{dataset}: already complete, reusing')
            continue
        groups = sorted({e['scene'] for e in entries})
        selected_groups = [groups[i] for i in np.linspace(0, len(groups) - 1, min(6, len(groups)), dtype=int)]
        selected_ids = {next(e['id'] for e in entries if e['scene'] == g) for g in selected_groups}
        results = defaultdict(list)
        start_time = time.monotonic()
        newly_measured = 0
        for start in range(0, len(entries), CHUNK):
            current = entries[start:start + CHUNK]
            ids = [e['id'] for e in current]
            shard = output / 'real' / '_parts' / f'seed{seed}' / dataset / f'{start:06d}.npz'
            if shard.exists():
                arrays = read_part(shard, provenance, ids)
            else:
                values = {f'{metric}_{row}': [] for row in ROWS for metric in ('psnr', 'ssim')}
                for e in current:
                    inp = load_gray(manifest.parent / e['input'])
                    target = load_gray(manifest.parent / e['target'])
                    if inp.shape != target.shape:
                        raise ValueError(f'{e["id"]}: different input and target shapes')
                    mask = None
                    if dataset == 'otis':
                        raw_mask = load_gray(manifest.parent / e['mask'])
                        if not np.isin(raw_mask, [0., 1.]).all():
                            raise ValueError('OTIS mask must be binary')
                        mask = raw_mask > 0
                    show = seed == 1 and e['id'] in selected_ids
                    images = {'input': inp, 'target': target} if show else None
                    for row in ROWS:
                        prediction = inp if row == 'noop' else predict_patches(
                            models[row], inp, int(cfg['data']['crop_px']), device,
                            int(cfg['train']['batch_size']))
                        scores = metric_scores(prediction, target, mask)
                        for metric, score in zip(('psnr', 'ssim'), scores):
                            values[f'{metric}_{row}'].append(score)
                        if show and row != 'noop':
                            images[row] = prediction
                    if show:
                        safe_id = hashlib.sha256(e['id'].encode()).hexdigest()[:12]
                        preview(output / 'previews' / f'{dataset}_{safe_id}.png', e['id'], images)
                arrays = dict(sample_id=np.array(ids), scene=np.array([e['scene'] for e in current]),
                              content_group=np.array([e.get('content_group', '') for e in current]),
                              **{k: np.asarray(v, np.float64) for k, v in values.items()})
                atomic_npz(shard, provenance=provenance, **arrays)
                newly_measured += len(current)
            for key in ('sample_id', 'scene', 'content_group', *[f'{m}_{r}' for r in ROWS for m in ('psnr', 'ssim')]):
                results[key].append(arrays[key])
            done = start + len(current)
            elapsed = time.monotonic() - start_time
            eta = (len(entries) - done) * elapsed / newly_measured if newly_measured else 0
            say(f'seed{seed}/{dataset}: {done}/{len(entries)} pairs; elapsed {elapsed/60:.1f} min; rough ETA {eta/60:.1f} min')
        atomic_npz(destination, provenance=provenance, **{k: np.concatenate(v) for k, v in results.items()})
        say(f'Saved {destination}')


def stats(values):
    a = np.asarray(values, np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Invalid values in summary')
    return dict(n=len(a), mean=float(a.mean()), min=float(a.min()), max=float(a.max()),
                range=float(np.ptp(a)))


def ratio(gain, denominator):
    return float(gain / denominator) if denominator > DENOM_EPS_DB else None


def group_means(values, groups):
    values, groups = np.asarray(values, np.float64), np.asarray(groups)
    return {str(g): float(values[groups == g].mean()) for g in np.unique(groups)}


def summarize(project, output):
    plan = json.loads((output / 'evaluation_plan.json').read_text())
    records, by_group, curves, training = [], [], [], []
    for seed in range(1, 5):
        cfg = plan['configs'][str(seed)]
        tag = Path(cfg['out_dir']).name.replace('experiments_', '')
        data = {}
        for row in ROWS:
            with np.load(output / 'synthetic' / f'{tag}__{row}.npz', allow_pickle=False) as f:
                data[row] = {k: f[k].copy() for k in f.files}
        provenance = str(data['noop']['provenance'])
        meta = json.loads(provenance)
        if meta['cfg'] != cfg or meta['checkpoints'] != plan['checkpoints'][str(seed)]:
            raise ValueError('Synthetic results do not match the frozen evaluation plan')
        if (meta['columns'] != list(LEVELS) or meta['dr_grid'] != list(STRENGTHS)
                or meta['physics_id'] != plan['physics_id'] or meta['protocol_id'] != plan['protocol_id']
                or meta['evaluator'] != plan['source_sha256']['src/eval_matrix.py']
                or meta['dataset'] != plan['synthetic_data_sha256']):
            raise ValueError('Incompatible synthetic evaluation provenance')
        for row in ROWS:
            if str(data[row]['provenance']) != provenance or not np.array_equal(data[row]['tile_idx'], data['noop']['tile_idx']):
                raise ValueError('Synthetic rows use different samples/provenance')
        for col in LEVELS:
            for strength in (*STRENGTHS, 'all'):
                strengths = STRENGTHS if strength == 'all' else [strength]
                means = {}
                for row in ROWS:
                    means[row] = {}
                    for metric in ('psnr', 'ssim'):
                        a = np.concatenate([data[row][f'{metric}_{col}_dr{v}'] for v in strengths])
                        if a.size != len(data[row]['tile_idx']) * len(strengths) or not np.isfinite(a).all():
                            raise ValueError('Invalid synthetic scores')
                        means[row][metric] = float(a.astype(np.float64).mean())
                add_records(records, seed, 'synthetic', col, str(strength), means, col)
        for dataset in REAL:
            path = output / 'real' / f'seed{seed}__{dataset}.npz'
            entries = read_manifest(project / REAL[dataset][0], dataset)
            expected = canonical(dict(plan=plan, seed=seed, dataset=dataset))
            real = read_part(path, expected, [e['id'] for e in entries])
            if list(real['scene']) != [e['scene'] for e in entries]:
                raise ValueError('Real result group assignment differs from manifest')
            means = {}
            for row in ROWS:
                means[row] = {}
                for metric in ('psnr', 'ssim'):
                    grouped = group_means(real[f'{metric}_{row}'], real['scene'])
                    means[row][metric] = float(np.mean(list(grouped.values())))
                    for scene, value in grouped.items():
                        by_group.append(dict(replica=seed, dataset=dataset, scene=scene,
                                             row=row, metric=metric, mean=value,
                                             frames=int((real['scene'] == scene).sum())))
            add_records(records, seed, 'real', dataset, 'all', means, 'd3')
            if dataset == 'otis':
                for chart in ('chart_A', 'chart_B'):
                    selected = real['content_group'] == chart
                    if not selected.any():
                        raise ValueError('Missing OTIS chart grouping')
                    chart_means = {row: {metric: float(np.mean(list(group_means(
                        real[f'{metric}_{row}'][selected], real['scene'][selected]).values())))
                        for metric in ('psnr', 'ssim')} for row in ROWS}
                    add_records(records, seed, 'real', 'otis_' + chart, 'all', chart_means, 'd3')
        for row in LEVELS:
            path = project / cfg['out_dir'] / row / 'log.csv'
            with path.open(newline='') as stream:
                log = list(csv.DictReader(stream))
            if [int(r['epoch']) for r in log] != list(range(1, 61)):
                raise ValueError(f'{path}: expected exactly epochs 1..60')
            for item in log:
                numeric = {k: float(item[k]) for k in ('step', 'lr', 'train_l1', 'val_psnr', 'val_ssim', 'sec')}
                if not all(np.isfinite(v) for v in numeric.values()):
                    raise ValueError(f'{path}: non-finite training log')
                curves.append(dict(replica=seed, row=row, epoch=int(item['epoch']), **numeric))
            val = np.array([float(r['val_psnr']) for r in log])
            training.append(dict(replica=seed, row=row, last_epoch=60,
                                 final_val_psnr=float(val[-1]), best_val_psnr=float(val.max()),
                                 best_val_epoch=int(val.argmax() + 1),
                                 final_minus_best_db=float(val[-1] - val.max()),
                                 last_10_val_span_db=float(np.ptp(val[-10:]))))
    aggregate = []
    for key in sorted({(r['domain'], r['column'], r['strength'], r['row']) for r in records}):
        group = sorted([r for r in records if (r['domain'], r['column'], r['strength'], r['row']) == key], key=lambda r: r['replica'])
        if [r['replica'] for r in group] != [1, 2, 3, 4]:
            raise ValueError('Summary requires exactly four paired replicas')
        record = dict(zip(('domain', 'column', 'strength', 'row'), key))
        for field in ('psnr_db', 'ssim', 'gain_db', 'gain_ssim', 'denominator_db', 'retained'):
            values = [r[field] for r in group]
            record[field] = None if any(v is None for v in values) else stats(values)
        aggregate.append(record)
    lookup = {(r['replica'], r['domain'], r['column'], r['strength'], r['row']): r for r in records}
    def cell(s, domain, col, strength, row):
        return lookup[(s, domain, col, str(strength), row)]
    effects = []
    ladder = LEVELS[:-1]
    for domain, col in [('synthetic', 'd3'), ('real', 'tsrwgan'), ('real', 'otis')]:
        for before, after in zip(ladder[:-1], ladder[1:]):
            for metric in ('psnr_db', 'ssim', 'retained'):
                values = []
                for s in range(1, 5):
                    a, b = cell(s, domain, col, 'all', before), cell(s, domain, col, 'all', after)
                    values.append(None if a[metric] is None or b[metric] is None else b[metric] - a[metric])
                effects.append(effect_record('ladder', domain, col, f'{after}-{before}', metric, values))
    for domain, col in [('synthetic', 'd1'), ('real', 'tsrwgan'), ('real', 'otis')]:
        for metric in ('psnr_db', 'ssim'):
            effects.append(effect_record('noise', domain, col, 'd3n-d3', metric, [
                cell(s, domain, col, 'all', 'd3n')[metric] - cell(s, domain, col, 'all', 'd3')[metric]
                for s in range(1, 5)]))
    interaction, retained_change = [], []
    for s in range(1, 5):
        def score(v, row):
            return cell(s, 'synthetic', 'd3', v, row)
        interaction.append((score(5, 'd3')['psnr_db'] - score(5, 'd1')['psnr_db'])
                           - (score(1, 'd3')['psnr_db'] - score(1, 'd1')['psnr_db']))
        den = score('all', 'd3')['denominator_db']
        retained_change.append(ratio(score(5, 'd1')['gain_db'] - score(1, 'd1')['gain_db'], den))
    effects.append(effect_record('strength_interaction', 'synthetic', 'd3',
                                 '(D3-D1)_dr5-(D3-D1)_dr1', 'psnr_db', interaction))
    effects.append(effect_record('strength_common_reference', 'synthetic', 'd3',
                                 '(gain_D1_dr5-gain_D1_dr1)/gain_D3_all', 'common_denominator_ratio', retained_change))
    robustness = []
    for s in range(1, 5):
        for row in LEVELS:
            gains = [cell(s, 'synthetic', col, 'all', row)['gain_db'] for col in CORE_COLUMNS]
            regrets = [max(cell(s, 'synthetic', col, 'all', r)['psnr_db'] for r in LEVELS)
                       - cell(s, 'synthetic', col, 'all', row)['psnr_db'] for col in CORE_COLUMNS]
            robustness.append(dict(replica=s, row=row, columns=','.join(CORE_COLUMNS),
                                   worst_gain_db=min(gains), max_regret_to_best_tested_db=max(regrets)))
    out = output / 'tables'
    csv_write(out / 'per_replica.csv', records)
    csv_write(out / 'real_per_recording.csv', by_group)
    csv_write(out / 'training_curves.csv', curves)
    csv_write(out / 'training_checks.csv', training)
    csv_write(out / 'robustness.csv', robustness)
    csv_write(out / 'paired_effects.csv', effects)
    flat = []
    for r in aggregate:
        item = {k: r[k] for k in ('domain', 'column', 'strength', 'row')}
        for field in ('psnr_db', 'ssim', 'gain_db', 'gain_ssim', 'denominator_db', 'retained'):
            for label in ('n', 'mean', 'min', 'max', 'range'):
                item[field + '_' + label] = None if r[field] is None else r[field][label]
        flat.append(item)
    csv_write(out / 'summary.csv', flat)
    summary = dict(replicas=4, per_replica=records, aggregate=aggregate, effects=effects,
                   robustness=robustness, training=training,
                   oracle_ceiling=dict(status='not_measured', reason='No held-out real-trained reference model exists in this campaign.'),
                   protocol=plan['evaluation_protocol'])
    json_write(output / 'summary.json', summary)
    write_report(output / 'REPORT_RU.md', summary)
    say(f'RESULTS COMPLETE: {output}/REPORT_RU.md and tables/')


def add_records(records, seed, domain, col, strength, means, reference):
    noop, ref = means['noop']['psnr'], means[reference]['psnr']
    den = ref - noop
    for row in ROWS:
        cell = means[row]
        records.append(dict(replica=seed, domain=domain, column=col, strength=strength, row=row,
                            psnr_db=cell['psnr'], ssim=cell['ssim'], noop_db=noop,
                            gain_db=cell['psnr'] - noop, gain_ssim=cell['ssim'] - means['noop']['ssim'],
                            reference_row=reference, denominator_db=den,
                            retained=ratio(cell['psnr'] - noop, den)))


def effect_record(question, domain, column, comparison, metric, values):
    valid = all(v is not None for v in values)
    st = stats(values) if valid else dict(n=len(values), mean=None, min=None, max=None, range=None)
    return dict(question=question, domain=domain, column=column, comparison=comparison,
                metric=metric, **st,
                absolute_mean_over_range=(abs(st['mean']) / st['range'] if valid and st['range'] > 0 else None),
                **{f'replica_{s}': v for s, v in enumerate(values, 1)})


def write_report(path, summary):
    def fmt(st, digits=3):
        if st is None:
            return 'не определено'
        return f'{st["mean"]:.{digits}f} [{st["min"]:.{digits}f}; {st["max"]:.{digits}f}]'
    text = ['# Итоговая оценка', '',
            'Четыре реплики. Числа: среднее [минимум; максимум] по репликам. '
            'Использованы только last.pt после 60 эпох; выбора чекпойнта по тесту нет.', '',
            'Синтетика: среднее по тестовым тайлам и пяти значениям D/r0. '
            'Реальные данные: сначала среднее по кадрам записи, затем равные веса записей. '
            'OTIS оценивается по маскам; TSRWGAN — по полному кадру. '
            'Датасеты не объединяются в один общий балл.', '']
    for domain, col in [('synthetic', v) for v in LEVELS] + [('real', 'tsrwgan'), ('real', 'otis')]:
        text.extend([f'## {domain}: {col}', '',
                     '| Модель | PSNR, дБ | SSIM | Прирост к no-op, дБ |',
                     '|---|---:|---:|---:|'])
        for row in ROWS:
            r = next(x for x in summary['aggregate'] if (x['domain'], x['column'], x['strength'], x['row']) == (domain, col, 'all', row))
            text.append(f'| {row} | {fmt(r["psnr_db"])} | {fmt(r["ssim"], 4)} | {fmt(r["gain_db"])} |')
        text.append('')
    text.extend(['## Что можно защищать', '',
        '1. Вклад шагов лестницы: paired_effects.csv, строки ladder. Сравнения парные внутри реплики; '
        'retained для синтетического D3 использует общий знаменатель D3 − no-op.',
        '2. Сила турбулентности: strength_interaction. Положительное значение означает, что '
        'преимущество D3 над D1 больше при D/r0=5, чем при 1. Это результат данной постановки, '
        'сам по себе он не опровергает чужую статью.',
        '3. Устойчивость к смене синтетики: robustness.csv. worst_gain_db — худший прирост к no-op '
        'по четырём заранее выбранным столбцам D0/D1/D2/D3; max_regret_to_best_tested_db — '
        'максимальное отставание от лучшей из проверенных моделей. Последний показатель описательный.',
        '4. Перенос на реальные съёмки: отдельные таблицы TSRWGAN и OTIS, парные шаги лестницы. '
        'TSRWGAN — проверенная подвыборка лабораторной физической турбулентности; OTIS — два типа '
        'мишеней в 16 последовательностях. Нельзя объявлять число кадров числом независимых сцен.',
        '5. Шум: строки noise сравнивают D3n − D3 отдельно на TSRWGAN, OTIS и синтетическом D1. '
        'Преимущество только на реальных данных совместимо с пользой моделирования шума, '
        'но не доказывает, что причина именно сенсорный шум.',
        '6. Доля от достижимого потолка не измерена: для неё нужен отдельный реальный обученный '
        'эталон с независимым тестом. D3 на реальных данных — только общая опорная модель, не потолок.', '',
        '## Ограничения интерпретации', '',
        '- retained вычислен как отношение агрегированных приростов PSNR, не как среднее отношений кадров. '
        'Знаменатель denominator_db всегда сохранён рядом в CSV. Знаменатель ≤ 0.000001 дБ даёт пустое значение '
        '(защита от численного нуля, не порог практической значимости); '
        'малый положительный знаменатель делает отношение нестабильным, поэтому основная таблица содержит также дБ.',
        '- Доли с разными знаменателями не вычитаются. Для изменения retained по силе предусмотрен '
        'отдельный общий знаменатель D3 − no-op, усреднённый по всем пяти силам.',
        '- abs(mean)/range описывает масштаб эффекта относительно наблюдаемого размаха четырёх реплик. '
        'Это не p-value, не доверительный интервал и не доказательство эквивалентности. При нулевом размахе отношение пустое.',
        '- Отрицательный перенос сохраняется в таблицах. Кадры по выходным метрикам сети не отбрасываются.',
        '- Цифровой GT OTIS может фотометрически отличаться от камерной съёмки. Без дополнительной подгонки яркости.',
        '- training_checks.csv показывает финальную валидацию, максимум за обучение и размах последних 10 эпох. '
        'Эти диагностики не заменяют анализа сходимости и не используются для выбора тестового чекпойнта.',
        '- previews/ содержит заранее выбранные примеры seed1 всех моделей. Уменьшение применяется только '
        'для отображения; метрики посчитаны в исходном разрешении.', '',
        '## Файлы', '',
        '- tables/summary.csv — среднее и размах каждой метрики по четырём репликам.',
        '- tables/per_replica.csv — агрегаты каждой реплики, включая знаменатель retained.',
        '- tables/paired_effects.csv — эффекты лестницы, силы и шума.',
        '- tables/real_per_recording.csv — результаты каждой реальной записи.',
        '- tables/robustness.csv — устойчивость по четырём синтетическим столбцам.',
        '- tables/training_curves.csv и training_checks.csv — обучение.',
        '- synthetic/ и real/ — исходные массивы метрик и части для возобновления.',
        '- evaluation_plan.json — контрольные суммы исходников, данных и весов.', ''])
    atomic_bytes(path, '\n'.join(text).encode())


def run(args):
    project = Path(args.project).resolve()
    os.chdir(project)
    sys.path.insert(0, str(project))
    output = Path(args.output).resolve()
    if args.worker is not None:
        return real_worker(project, output, args.worker)
    if args.summarize_only:
        return summarize(project, output)
    import torch
    import check_time_8gpu as common
    from src.physics_version import physics_id, protocol_id
    from src.distortion import LEVELS as registry
    if common.VERSION != 2 or tuple(registry) != LEVELS:
        raise RuntimeError('Unexpected campaign launcher/level registry version')
    torch.set_num_threads(1)
    lock = common.acquire_lock(project)
    common.install_signals()
    try:
        say('Final evaluation v1: checking completed training and prepared datasets')
        configs = common.checked_configs(project)
        common.check_level_locks(project, configs)
        steps = common.training_steps(project, configs)
        checkpoints = {}
        for seed, cfg in configs.items():
            checkpoints[str(seed)] = {}
            for level in LEVELS:
                path = project / cfg['out_dir'] / level / 'last.pt'
                epoch = common.checkpoint_epoch(path, cfg, level, steps_per_epoch=steps[seed])
                if epoch != 60:
                    raise RuntimeError(f'seed{seed}/{level}: epoch {epoch}/60; wait for training completion')
                with (path.parent / 'log.csv').open(newline='') as stream:
                    logged = list(csv.DictReader(stream))
                if [int(r['epoch']) for r in logged] != list(range(1, 61)):
                    raise RuntimeError(f'{path.parent}/log.csv: expected all 60 completed epochs')
                if not all(np.isfinite(float(r[k])) for r in logged
                           for k in ('step', 'lr', 'train_l1', 'val_psnr', 'val_ssim', 'sec')):
                    raise RuntimeError(f'{path.parent}/log.csv: non-finite training statistics')
                checkpoints[str(seed)][level] = sha(path)
            say(f'seed{seed}: all six final checkpoints verified')
        metric_selftest()
        real_sources = {dataset: verify_sources(project, dataset) for dataset in REAL}
        source_names = ['src/eval_matrix.py', 'src/eval_real.py', 'src/metrics.py', 'check_time_8gpu.py']
        data = configs[1]['data']
        plan = dict(version=VERSION, configs={str(s): c for s, c in configs.items()},
                    checkpoints=checkpoints, physics_id=physics_id(), protocol_id=protocol_id(),
                    driver_sha256=sha(__file__), source_sha256={p: sha(project / p) for p in source_names},
                    synthetic_data_sha256={p: sha(project / data['tiles'] / p) for p in ('source_id.npy', 'tiles.npy')},
                    real_sources=real_sources,
                    runtime={p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scipy', 'scikit-image', 'Pillow')},
                    evaluation_protocol=dict(
                        checkpoint='last.pt, epoch 60; four replicas',
                        synthetic_columns=list(LEVELS), strengths=list(STRENGTHS),
                        synthetic_aggregation='mean per tile, then equal weights for five strengths',
                        real_aggregation='mean per frame within recording, then equal weights for recordings',
                        real_inference='float32; training-size patches; 50% overlap; uniform averaging; reflect padding if small',
                        real_registration='none', masked_psnr='mean squared error over valid pixels',
                        masked_ssim='Wang Gaussian 11x11 sigma1.5 population covariance; fully valid windows only',
                        real_retained_reference='D3, not a real-trained oracle',
                        denominator_numerical_zero_db=DENOM_EPS_DB, model_dependent_filtering=False))
        path = output / 'evaluation_plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise RuntimeError('Evaluation inputs/protocol changed; existing outputs preserved. Use a fresh --output directory.')
        if args.check_only:
            say('CHECK OK: all 24 models complete; real datasets verified; no evaluation started')
            return
        if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
            raise RuntimeError('This launcher expects eight visible CUDA GPUs')
        json_write(path, plan)
        entry = "import runpy,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; torch.set_num_threads(1); torch.cuda.set_device(0); runpy.run_module('src.eval_matrix',run_name='__main__')"
        jobs = []
        for seed in range(1, 5):
            jobs.append(dict(name=f'seed{seed}/synthetic', gpu=2 * (seed - 1),
                             command=[sys.executable, '-u', '-c', entry, '--config', f'configs/seed{seed}.yaml',
                                      '--output', str(output / 'synthetic')],
                             log=output / 'logs' / f'seed{seed}_synthetic.log'))
            jobs.append(dict(name=f'seed{seed}/real', gpu=2 * (seed - 1) + 1,
                             command=[sys.executable, '-u', str(Path(__file__).resolve()), '--project', str(project),
                                      '--output', str(output), '--worker', str(seed)],
                             log=output / 'logs' / f'seed{seed}_real.log'))
        start = time.monotonic()
        json_write(output / 'status.json', dict(status='running', pid=os.getpid()))
        say('Starting 4 synthetic matrices + 4 real-data workers on 8 GPUs')
        def progress(names):
            say(f'Active ({(time.monotonic()-start)/3600:.2f} h): ' + ', '.join(names))
        common.parallel_jobs(jobs, project, progress, lock=lock)
        summarize(project, output)
        json_write(output / 'status.json', dict(status='complete', elapsed_seconds=time.monotonic()-start,
                                               replicas=4, models=24, output=str(output)))
        say('COMPLETE. Evaluation finished. Save results AND trained checkpoints before removing the instance.')
    except BaseException as exc:
        if not args.check_only:
            json_write(output / 'status.json', dict(status='failed', error=str(exc)))
        raise
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project', default='.')
    p.add_argument('--output', default='results_final')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--summarize-only', action='store_true')
    modes.add_argument('--worker', type=int, choices=range(1, 5), help=argparse.SUPPRESS)
    args = p.parse_args()
    try:
        run(args)
    except (Exception, KeyboardInterrupt) as exc:
        say(f'ERROR: {exc}')
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
