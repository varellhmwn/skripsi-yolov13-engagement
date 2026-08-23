"""
process_hard_samples_by_subject.py — Entry Point
"""
import sys
from pathlib import Path
from scripts.process_hard_samples_by_subject import parse_arguments, process_hard_samples

if __name__ == '__main__':
    args = parse_arguments()

    mapping_path = Path(args.mapping_dir)
    master_path = Path(args.master_dir)
    output_path = Path(args.output_dir)

    if not mapping_path.exists() and (Path('datasets') / args.mapping_dir).exists():
        mapping_path = Path('datasets') / args.mapping_dir
    if not master_path.exists() and (Path('datasets') / args.master_dir).exists():
        master_path = Path('datasets') / args.master_dir

    try:
        process_hard_samples(
            mapping_dir=mapping_path,
            master_dir=master_path,
            output_dir=output_path,
            dry_run=args.dry_run
        )
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
