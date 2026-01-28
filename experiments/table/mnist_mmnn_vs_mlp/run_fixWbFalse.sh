#!/bin/bash
# we run MMNN with fixWb=False for R in 5,10,15,25,50; first layer factorized 784→128→512
# run from project root:  bash experiments/table/mnist_mmnn_vs_mlp/run_fixWbFalse.sh

set -e

# param counts with --factorize-first 128: total = 172042 + 1025*R; fixWb=False => trainable = total
echo "MMNN fixWb=False, factorize-first=128 (Params = Trainable = 172,042 + 1,025*R):"
echo ""
printf "| R   | Params   | Trainable |\n"
printf "|-----|----------|----------|\n"
printf "|  5  | 177,167  | 177,167  |\n"
printf "| 10  | 182,292  | 182,292  |\n"
printf "| 15  | 187,417  | 187,417  |\n"
printf "| 25  | 197,667  | 197,667  |\n"
printf "| 50  | 223,292  | 223,292  |\n"
echo ""
echo "running training (--no-fix-wb --factorize-first 128 --mmnn-ranks 5 10 15 25 50 --skip-mlp)..."
echo ""

python experiments/table/mnist_mmnn_vs_mlp.py \
  --no-fix-wb \
  --factorize-first 128 \
  --mmnn-ranks 5 10 15 25 50 \
  --skip-mlp \
  --out-dir experiments/table/mnist_mmnn_vs_mlp_fixWbFalse

echo ""
echo "done. results in experiments/table/mnist_mmnn_vs_mlp_fixWbFalse/ (results.json, mmnn_r*.pt, ...)"
