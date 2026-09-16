"""One independent training process per GPU; resume completed epochs safely."""
import argparse
from pathlib import Path
import subprocess
import sys
import yaml
from src.distortion import LEVELS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=range(1, 6), required=True)
    p.add_argument('--levels', nargs='+', choices=list(LEVELS), default=list(LEVELS))
    p.add_argument('--evaluate', action='store_true')
    a = p.parse_args()
    config = Path(f'configs/seed{a.seed}.yaml')
    cfg = yaml.safe_load(config.read_text())
    for level in a.levels:
        command = [sys.executable, '-u', '-m', 'src.train', '--config', str(config), '--level', level]
        if (Path(cfg['out_dir']) / level / 'last.pt').exists():
            command.append('--resume')
        print(' '.join(command), flush=True)
        subprocess.run(command, check=True)
    if a.evaluate:
        subprocess.run([sys.executable, '-u', '-m', 'src.eval_matrix', '--config', str(config)], check=True)


if __name__ == '__main__':
    main()
