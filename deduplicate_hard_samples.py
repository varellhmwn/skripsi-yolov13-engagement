"""
deduplicate_hard_samples.py — Entry Point
"""
import sys
from pathlib import Path
from scripts.deduplicate_hard_samples import parse_args, deduplicate_dataset

if __name__ == '__main__':
    args = parse_args()
    in_path = Path(args.input_dir)
    out_path = Path(args.output_dir)

    try:
        deduplicate_dataset(
            input_dir=in_path,
            output_dir=out_path,
            dry_run=args.dry_run
        )
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
