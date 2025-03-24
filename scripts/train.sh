#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools

mkdir -p $models

num_threads=4
device="0"

SECONDS=0

dropouts=(0.0, 0.25, 0.5, 0.75, 1.0)

for dropout in "${dropouts[@]}"; do
    (cd $tools/pytorch-examples/word_language_model &&
        CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/shelley \
            --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout $dropout --tied \
        --save $models/model.pt \
        --cuda \
)

echo "time taken:"
echo "$SECONDS seconds"
