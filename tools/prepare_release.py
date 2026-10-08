"""Build a source-only CGTPTrack ZIP; never include local datasets or credentials."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
TOP = {'.gitignore', 'README.md', 'LICENSE', 'install.sh', 'requirements.txt', 'run_test.py'}
TRACKING = {'_init_paths.py', 'train.py', 'test.py', 'analysis_results.py', 'create_default_local_file.py'}
EXCLUDE = {
    'lib/train/admin/local.py', 'lib/test/evaluation/local.py',
    'lib/test/tracker/cgtptrack-0.py', 'lib/test/tracker/cgtptrack2.py',
    'lib/test/tracker/cgtptrack_fallback.py',
    'lib/test/parameter/cgtptrack_fallback.py', 'lib/test/parameter/cgtptrack_t6orig.py',
}


def source_files():
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part.startswith('.') or part == '__pycache__' for part in rel.parts[:-1]):
            continue
        if rel.as_posix() in EXCLUDE:
            continue
        if (len(rel.parts) == 1 and path.name in TOP
                or rel.parts[0] == 'lib' and path.suffix in {'.py', '.txt', '.md'}
                or rel.parts[0] == 'experiments' and path.suffix == '.yaml'
                or rel.parts[0] in {'tests', 'tools'} and path.suffix == '.py'
                or rel.parts[0] == 'tracking' and path.name in TRACKING
                or rel.as_posix() in {'docs/paper-alignment.md', 'docs/validation.txt'}):
            yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'CGTPTrack_github.zip')
    args = parser.parse_args()
    with ZipFile(args.output, 'w', ZIP_DEFLATED) as archive:
        for path in source_files():
            archive.write(path, 'CGTPTrack/' + path.relative_to(ROOT).as_posix())
    with ZipFile(args.output) as archive:
        assert archive.testzip() is None
        print(f'{args.output}: {len(archive.namelist())} source files')


if __name__ == '__main__':
    main()
