"""Evaluate CGTPTrack result files; run from the repository root."""
import argparse
import _init_paths
from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='lasot')
    parser.add_argument('--config', default='baseline_full', choices=['baseline', 'baseline_full'])
    parser.add_argument('--runid', type=int, default=None)
    args = parser.parse_args()
    trackers = trackerlist('cgtptrack', args.config, args.dataset,
                           run_ids=args.runid, display_name='CGTPTrack')
    print_results(trackers, get_dataset(args.dataset), args.dataset,
                  merge_results=True, plot_types=('success', 'norm_prec', 'prec'))


if __name__ == '__main__':
    main()
