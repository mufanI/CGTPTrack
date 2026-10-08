"""Launch the two CGTPTrack training configurations with safe argument passing."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description='Train CGTPTrack')
    parser.add_argument('--script', choices=['cgtptrack'], default='cgtptrack')
    parser.add_argument('--config', choices=['baseline', 'baseline_full'], default='baseline_full')
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--mode', choices=['single', 'multiple'], default='single')
    parser.add_argument('--nproc_per_node', type=int, default=1)
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable]
    if args.mode == 'multiple':
        command += ['-m', 'torch.distributed.run', '--standalone',
                    '--nproc_per_node', str(args.nproc_per_node)]
    command += [str(root / 'lib/train/run_training.py'), '--script', args.script,
                '--config', args.config, '--save_dir', str(Path(args.save_dir).resolve()),
                '--use_lmdb', str(args.use_lmdb), '--use_wandb', str(args.use_wandb)]
    subprocess.run(command, cwd=str(root), check=True)


if __name__ == '__main__':
    main()
