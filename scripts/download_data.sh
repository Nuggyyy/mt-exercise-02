#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!

mkdir -p $data/shelley

mkdir -p $data/shelley/raw

curl -O https://www.gutenberg.org/cache/epub/84/pg84.txt
mv pg84.txt $data/shelley/raw/frankenstein.txt

# preprocess slightly

cat $data/shelley/raw/frankenstein.txt | python $base/scripts/preprocess_raw.py > $data/shelley/raw/frankenstein.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/shelley/raw/frankenstein.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/shelley/raw/frankenstein.preprocessed.txt

# split into train, valid and test

head -n 440 $data/shelley/raw/frankenstein.preprocessed.txt | tail -n 400 > $data/shelley/valid.txt
head -n 840 $data/shelley/raw/frankenstein.preprocessed.txt | tail -n 400 > $data/shelley/test.txt
tail -n 3075 $data/shelley/raw/frankenstein.preprocessed.txt | head -n 2955 > $data/shelley/train.txt
