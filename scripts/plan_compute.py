"""Schedule 30 independent training jobs from measured full-pipeline timings.
No DDP, batch-size changes, hardware scaling guesses, or extrapolation of
old simulator timings. Evaluation is a separate conservative upper bound.
"""
import argparse
import json
import math
from pathlib import Path
from src.distortion import LEVELS
from src.physics_version import physics_id


def schedule(durations, gpus):
    loads = [0.] * gpus
    jobs = [[] for _ in loads]
    for name, duration in sorted(durations, key=lambda x: -x[1]):
        i = min(range(gpus), key=lambda k: loads[k])
        loads[i] += duration
        jobs[i].append(name)
    return max(loads), jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('measurements', nargs='+', type=Path)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--reserve', type=float, default=.2)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    measured = {x['level']: x for x in (json.loads(f.read_text()) for f in a.measurements)}
    missing = set(LEVELS) - measured.keys()
    if missing:
        p.error(f'missing timings: {sorted(missing)}')
    if a.epochs < 1 or a.reserve < 0:
        p.error('invalid epochs/reserve')
    if any(x['physics_id'] != physics_id() for x in measured.values()):
        p.error('timings are for a different simulator version')
    if len({(x['gpu'], x['batch_size'], x['workers']) for x in measured.values()}) != 1:
        p.error('measure all levels on the same GPU, batch size and worker count')
    factor = 1 + a.reserve
    durations = [(f's{s}/{lv}', x['epoch_estimate_seconds'] * a.epochs / 3600 * factor)
                 for s in range(1, 6) for lv, x in measured.items()]
    # Five strength values per column; synthesis once, six models plus no-op.
    eval_one = sum(5 * x['test_batches'] *
                   (x['val_synthesis_seconds'] + 7 * max(0, x['val_step_seconds'] - x['val_synthesis_seconds']))
                   for x in measured.values()) / 3600 * factor
    rows = []
    print('GPU | training h | synthetic evaluation h | conservative total h')
    for g in (3, 5, 6, 8, 10, 15, 30):
        training, jobs = schedule(durations, g)
        evaluation = math.ceil(5 / g) * eval_one
        row = dict(gpus=g, training_hours=training, synthetic_eval_hours=evaluation,
                   total_hours=training+evaluation, assignments=jobs)
        rows.append(row)
        print(f'{g:3d} | {training:10.2f} | {evaluation:22.2f} | {training+evaluation:20.2f}')
    result = dict(epochs=a.epochs, gpu_hours=sum(d for _, d in durations),
                  longest_run_hours=max(d for _, d in durations), reserve=a.reserve,
                  note='Same GPU type/CPU allocation as benchmark. Excludes transfers, real evaluation and real-data ceiling training. Synthetic eval upper bound assumes seven full forwards; no-op actually needs no forward.',
                  schedules=rows)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
