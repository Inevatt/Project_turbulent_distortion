#!/usr/bin/env python3
"""SERVER ONLY: train seed1..seed4, six levels, 60 epochs on eight RTX 5090.

Put this file and check_time_8gpu.py in the project root. Activate venv.
First run: python -u check_time_8gpu.py
Then run:  nohup python -u train_8gpu.py >> campaign_train.log 2>&1 &

No installation, transfer, benchmark, automatic shutdown or epoch changes.
Resumes each level from its last completed epoch. A second concurrent launch
is refused. Each worker log is under campaign_8gpu/logs/.
Evaluation is optional: add --evaluate to run synthetic matrices afterwards.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback

try:
    import check_time_8gpu as common
except ImportError as exc:
    raise SystemExit('Place check_time_8gpu.py beside train_8gpu.py') from exc

if common.VERSION != 2:
    raise SystemExit('Download the current pair of check_time_8gpu.py and train_8gpu.py together')

# Check CUDA inside EVERY training subprocess: src.train otherwise falls back
# to CPU when CUDA is unavailable, which would invalidate the time estimate.
TRAIN_ENTRY = """
import runpy, torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Expected exactly one usable CUDA GPU; refusing CPU training')
torch.cuda.set_device(0)
torch.empty(1, device='cuda:0')
runpy.run_module('src.train', run_name='__main__')
"""
EVAL_ENTRY = TRAIN_ENTRY.replace('src.train', 'src.eval_matrix')


def run_queue(args):
    project = common.project_root(args.project)
    cfg = common.checked_configs(project)[args.queue_seed]
    fd = int(os.environ['TURBDIST_LOCK_FD'])
    for level in args.queue_levels:
        command = [sys.executable, '-u', '-c', TRAIN_ENTRY,
                   '--config', f'configs/seed{args.queue_seed}.yaml', '--level', level]
        if (project/cfg['out_dir']/level/'last.pt').exists():
            command.append('--resume')
        common.say(f'seed{args.queue_seed}: starting {level}')
        # Retain the campaign lock if the supervisor/wrapper is killed.
        common.execute(command, cwd=project, pass_fds=(fd,))
    return 0


def validate_report(report, configs, hardware):
    if report.get('status') != 'complete':
        raise RuntimeError('Timing check is incomplete/failed; inspect time_check_8gpu/summary.json')
    for key, value in [('version', common.VERSION), ('physics_id', common.PHYSICS), ('protocol_id', common.PROTOCOL),
                       ('configs', json.loads(json.dumps(configs)))]:
        if report.get(key) != value:
            raise RuntimeError(f'{key} changed after timing check; run check_time_8gpu.py again')
    for key in ('inputs', 'gpus', 'driver', 'cpu_affinity', 'cpu_max', 'torch', 'cuda', 'visibility', 'runtime'):
        if report['hardware'].get(key) != hardware.get(key):
            raise RuntimeError(f'Hardware/data field {key} changed; run check_time_8gpu.py again')


def run_training(args):
    project = common.project_root(args.project)
    lock = common.acquire_lock(project)
    common.install_signals()
    root = project/'campaign_8gpu'
    def status(stage, **extra):
        common.atomic_json(root/'status.json', dict(stage=stage, pid=os.getpid(),
                                                    updated=common.stamp(), **extra))
    try:
        common.say(f'Training launcher version {common.VERSION}')
        status('checking')
        report_path = project/'time_check_8gpu/summary.json'
        if not report_path.exists():
            raise RuntimeError('First run python -u check_time_8gpu.py and review its forecast')
        report = json.loads(report_path.read_text())
        configs = common.checked_configs(project)
        common.check_level_locks(project, configs)
        hardware = common.hardware_and_data(project)
        validate_report(report, configs, hardware)
        manifest = json.loads(json.dumps(dict(configs=configs, inputs=hardware['inputs'],
                                              physics_id=common.PHYSICS, protocol_id=common.PROTOCOL)))
        manifest_path = root/'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Existing campaign uses different inputs/configs; checkpoints preserved')
        common.atomic_json(manifest_path, manifest)
        steps_by_seed = common.training_steps(project, configs)
        remaining = {}
        for seed,cfg in configs.items():
            for level in common.LEVELS:
                epoch = common.checkpoint_epoch(project/cfg['out_dir']/level/'last.pt', cfg, level,
                                                steps_per_epoch=steps_by_seed[seed])
                remaining[(seed,level)] = 60-epoch
                if 0 < epoch < 60:
                    common.repair_resume_log(project/cfg['out_dir']/level/'log.csv', epoch)
        measurements = {(x['slot'], x['level']):x for x in report['measurements']}
        forecast = common.make_forecast(measurements, remaining, args.evaluate)
        common.say(f'Estimated remaining training with 20% reserve: '
                   f'{forecast["training_hours_with_reserve"]:.2f} h')
        if args.evaluate:
            common.say(f'Synthetic evaluation afterwards: ~{forecast["synthetic_eval_hours_heuristic"]:.2f} h extra')
        common.say('TRAINING: four replicas, six levels, original 60-epoch configs; no timing rerun')
        jobs = []
        for q in common.queues():
            seed,gpu = q['seed'],q['gpu']
            levels = [l for l in q['levels'] if remaining[(seed,l)] > 0]
            if levels:
                jobs.append(dict(name=f'gpu{gpu}/seed{seed}', gpu=gpu,
                    command=[sys.executable, '-u', str(Path(__file__).resolve()), '--project', str(project),
                             '--queue-seed', str(seed), '--queue-levels', *levels],
                    env={'TURBDIST_LOCK_FD':str(lock.fileno())},
                    log=root/'logs'/f'gpu{gpu}_seed{seed}.log'))
        def progress(names):
            epochs = {}
            for seed,cfg in configs.items():
                for level in common.LEVELS:
                    f = project/cfg['out_dir']/level/'log.csv'
                    if f.exists():
                        try:
                            with f.open() as stream:
                                rows = list(csv.DictReader(stream))
                            if rows:
                                epochs[f's{seed}/{level}'] = int(rows[-1]['epoch'])
                        except (ValueError, KeyError, TypeError):
                            pass  # A writer can still be completing the last CSV row.
            status('training', active=names, logged_epochs=epochs, forecast=forecast)
            common.say('Active: '+', '.join(names)+'; epochs: '+json.dumps(epochs))
        status('training', forecast=forecast)
        common.parallel_jobs(jobs, project, progress, lock=lock)
        for seed,cfg in configs.items():
            for level in common.LEVELS:
                if common.checkpoint_epoch(project/cfg['out_dir']/level/'last.pt',cfg,level,
                                           steps_per_epoch=steps_by_seed[seed]) != 60:
                    raise RuntimeError(f'Incomplete checkpoint: seed{seed}/{level}')
        if args.evaluate:
            status('evaluation')
            common.say('Training complete. Running four synthetic matrices.')
            jobs = [dict(name=f'eval_seed{s}',gpu=(s-1)*2,
                command=[sys.executable,'-u','-c',EVAL_ENTRY,'--config',f'configs/seed{s}.yaml'],
                log=root/'logs'/f'eval_seed{s}.log') for s in range(1,5)]
            common.parallel_jobs(jobs,project,lambda names: common.say('Evaluation: '+', '.join(names)),lock=lock)
            common.execute([sys.executable,'-u','-m','scripts.summarize_results','--configs',
                            *[f'configs/seed{s}.yaml' for s in range(1,5)]],cwd=project)
        status('complete',replicas=4,models=24,synthetic_evaluation=args.evaluate)
        common.say('COMPLETE: all 24 models reached 60 epochs. Checkpoints: experiments_audit_v1_s1..s4/')
        common.say('Instance is still running. Stop it manually in Vast after saving your results.')
        return 0
    except BaseException as e:
        status('stopped' if isinstance(e,(InterruptedError,KeyboardInterrupt)) else 'failed',error=str(e))
        raise
    finally:
        lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project',default='.',help='Project root; default current directory')
    p.add_argument('--evaluate',action='store_true',help='Also run final synthetic matrices and four-replica summary')
    p.add_argument('--queue-seed',type=int,choices=range(1,5),help=argparse.SUPPRESS)
    p.add_argument('--queue-levels',nargs='+',choices=common.LEVELS,help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.queue_seed is not None:
        if not args.queue_levels or 'TURBDIST_LOCK_FD' not in os.environ:
            p.error('Internal queue mode must be started by the campaign supervisor')
        return run_queue(args)
    return run_training(args)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (Exception,KeyboardInterrupt) as exc:
        common.say(f'ERROR: {exc}')
        traceback.print_exc()
        sys.exit(1)
