# MT Exercise 2: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/marpng/mt-exercise-02
    cd mt-exercise-02

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh



# Changes Made

## make_virtualenv.sh
- python3 -> python

## download_data.sh
- wget -> curl -O
- changed all "grimm" to "shelley"
- changed all "tales" to "frankenstein"
- changed link to the correct .txt file to download

## preprocess.py
- added nltk.download('punkt_tab')

## train.sh
- made pytorch use my GPU (device="0"/--cuda)
- changed "grimm" to "shelley"

## data.py in torch
- changed the encoding to latin-1 in line 31 and 38

## main.py in torch
- changed line 246 model = model.load(f, weights_only=False)

## generate.sh
- made pytorch use my GPU (device="0"/--cuda)
- changed "grimm" to "shelley"

## generate.py in torch
- specified encoding in line 66 to latin-1
- changed line 55 model = torch.load(f, map_location=device, weights_only=False)

## logging
- you can see copy_of_main.py for changes
- running the script traintask2.sh should train 5 language models with dropout rates [0, 0.25, 0.5, 0.75, 1]
- it will also create a test-, train-, and valid_dropout_{dropout_rate}.csv file for each language model in the /metrics folder
- when done iterating over the dropout rates, it will then consolidate all the .csv in /metrics into one test.csv, train.csv and valid.csv in the /logs folder
- then, it will create a line plot for the train and valid perplexities and save it as perplexity_results.png in the /logs folder
- additionally, all trained models are also available as .pt files in the /models folder

# Commands to run to replicate task 2
- you'll need to pip install additional libraries which would be
```
pip install pandas seaborn matplotlib numpy
```
- For the first part about logging and visualising data, you only need to run
```
./scripts/traintask2.sh
```
- To generate the text from the highest and lowest perplexity models (dropout 1.0 and dropout 0.0 respectively), run
```
./scripts/generate_highest.sh
```
- and
```
./scripts/generate_lowest.sh
```
