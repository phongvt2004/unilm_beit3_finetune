path=$1

shift 1 # Shift the first 5 arguments, so $1 now refers to the 6th argument

for step in "$@"; do
  echo "Merging predictions for step ================================== $step"
  python scripts/math_scale/merge_dp_predictions.py --input_file "$path/mwpbench/checkpoint-$step/train_wo_gsm.2k.v1.0.0shot.n1.tem0.0.p1.0.?-of-8.json" \
    --output_file "$path/mwpbench/checkpoint-$step/train_wo_gsm.2k.v1.0.0shot.n1.tem0.0.p1.0.json"
  python scripts/math/merge_dp_predictions.py --input_file "$path/math/checkpoint-$step/math.test.v1.1.0shot.n1.tem0.0.p1.0.8-of-?.json" \
    --output_file "$path/math/checkpoint-$step/math.test.v1.1.0shot.n1.tem0.0.p1.0.json"
  echo "Done merging predictions for step ================================== $step"
  echo
done