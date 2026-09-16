"""Raw metrics and retained, with each denominator reported explicitly.
Retained is a ratio of dataset mean PSNR gains, never a mean of per-image
ratios. A nonpositive denominator is invalid, not clipped or made positive.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from src.distortion import LEVELS


def retained(cell, noop, reference):
    denominator = float(reference - noop)
    if not np.isfinite(denominator) or denominator <= 1e-8:
        return None, denominator
    return float((cell - noop) / denominator), denominator


def stats(values):
    v = np.asarray(values, dtype=float)
    return dict(n=len(v), mean=float(v.mean()), min=float(v.min()),
                max=float(v.max()), range=float(np.ptp(v)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', nargs='+', default=[f'configs/seed{s}.yaml' for s in range(1, 6)])
    p.add_argument('--results', type=Path, default=Path('results'))
    p.add_argument('--output', type=Path, default=Path('results/summary.json'))
    a = p.parse_args()
    records = []
    for config in a.configs:
        cfg = yaml.safe_load(Path(config).read_text())
        tag = Path(cfg['out_dir']).name.replace('experiments_', '')
        data = {}
        for row in ('noop', *LEVELS):
            with np.load(a.results / f'{tag}__{row}.npz', allow_pickle=False) as f:
                data[row] = {k: f[k] for k in f.files}
        for row in LEVELS:
            if str(data[row]['provenance']) != str(data['noop']['provenance']):
                raise ValueError(f'{tag}: mixed provenance')
        for col in LEVELS:
            keys = sorted(k for k in data['noop'] if k.startswith(f'psnr_{col}_dr'))
            if not keys:
                continue
            for strength in [k.split('_dr')[1] for k in keys] + ['all']:
                use = keys if strength == 'all' else [f'psnr_{col}_dr{strength}']
                def mean(row, metric='psnr'):
                    return float(np.concatenate([data[row][k.replace('psnr_', metric+'_', 1)] for k in use]).mean())
                noop, diag = mean('noop'), mean(col)
                for row in LEVELS:
                    cell = mean(row)
                    r, den = retained(cell, noop, diag)
                    records.append(dict(seed=cfg['seed'], row=row, column=col, strength=strength,
                                        psnr_db=cell, ssim=mean(row, 'ssim'), noop_db=noop,
                                        denominator_db=den, retained=r))
    summaries = []
    for key in sorted({(r['row'], r['column'], r['strength']) for r in records}):
        selected = [r for r in records if (r['row'], r['column'], r['strength']) == key]
        summaries.append(dict(row=key[0], column=key[1], strength=key[2],
                              psnr_db=stats([r['psnr_db'] for r in selected]),
                              ssim=stats([r['ssim'] for r in selected]),
                              denominator_db=stats([r['denominator_db'] for r in selected]),
                              retained=stats([r['retained'] for r in selected]) if all(r['retained'] is not None for r in selected) else None))
    # Paired adjacent ladder effects on ONE column with ONE denominator per seed.
    ladder = []
    levels = ['d0', 'd1', 'd15', 'd2', 'd3']
    for before, after in zip(levels[:-1], levels[1:]):
        selected = [r for r in records if r['column']=='d3' and r['strength']=='all']
        values = []
        for seed in sorted({r['seed'] for r in selected}):
            s = {r['row']: r for r in selected if r['seed']==seed}
            if s[before]['retained'] is not None and s[after]['retained'] is not None:
                values.append(s[after]['retained']-s[before]['retained'])
        if values:
            st = stats(values)
            ladder.append(dict(before=before, after=after, paired_delta=st,
                               absolute_mean_over_range=abs(st['mean'])/st['range'] if st['range']>0 else None))
    result = dict(per_replica=records, summaries=summaries, ladder_d3=ladder,
                  interpretation='No p-values or ROPE. Diagonal is a reference, not a guaranteed maximum. Strength-specific retained values have different denominators; do not subtract them without defining a common reference. Real-data retained requires an explicit reference model.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(a.output)


if __name__ == '__main__':
    main()
