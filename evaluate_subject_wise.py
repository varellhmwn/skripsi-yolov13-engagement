"""
evaluate_subject_wise.py — Root Entry Point
"""
import sys
from scripts.evaluate_subject_wise import parse_args, run_evaluation

if __name__ == '__main__':
    args = parse_args()
    try:
        run_evaluation(args)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
