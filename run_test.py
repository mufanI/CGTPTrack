"""Convenience entry point for CGTPTrack evaluation without rewriting configs."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='baseline_full', choices=['baseline', 'baseline_full'])
    parser.add_argument('--dataset', default='lasot')
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(root / 'tracking/test.py'), 'cgtptrack', args.config,
                    '--dataset_name', args.dataset, '--threads', str(args.threads),
                    '--num_gpus', str(args.num_gpus)], cwd=str(root), check=True)


if __name__ == '__main__':
    main()
