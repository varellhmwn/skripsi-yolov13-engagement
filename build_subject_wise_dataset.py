"""
build_subject_wise_dataset.py — Entry Point
"""
import sys
from pathlib import Path
from scripts.build_subject_wise_dataset import parse_args, build_dataset_pipeline

if __name__ == '__main__':
    args = parse_args()
    pub_p = Path(args.public_dir)
    priv_p = Path(args.private_dir)
    out_p = Path(args.output_dir)

    try:
        build_dataset_pipeline(
            public_dir=pub_p,
            private_dir=priv_p,
            output_dir=out_p,
            dry_run=args.dry_run,
            candidate_idx=args.candidate
        )
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
