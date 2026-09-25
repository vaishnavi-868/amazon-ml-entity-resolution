#!/usr/bin/env bash
# End-to-end: train -> infer. Run from code/business_entity_resolution/
# Usage: bash scripts/run_all.sh /path/to/student_resource/dataset /path/to/output [--no-dense]
set -euo pipefail
DATA=${1:-dataset}; OUT=${2:-output}; shift 2 || true
python -m src.train --data-dir "$DATA" --work-dir work --model-dir models --n-jobs 4 "$@"
python -m src.infer --data-dir "$DATA" --work-dir work --model-dir models --out-dir "$OUT" --n-jobs 4
echo
echo "Now run the official validator from student_resource/:"
echo "  python3 utils/validate_submission.py --matching $OUT/matching_results.tsv \\"
echo "      --candidate $OUT/candidate_pairs.tsv --test-dir $DATA/test"
