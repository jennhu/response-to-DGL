# Language models align with humans' judgments on key grammatical constructions

This repository contains code and data for a response to the paper
["Systematic testing of three Language Models reveals low language accuracy, absence of response stability, and a yes-response bias"](https://www.pnas.org/doi/10.1073/pnas.2309583120) (Dentella et al, *PNAS* 2023; "DGL").

## Dependencies

To reproduce our figures/analyses, you will only need basic scientific Python
and visualization tools (`matplotlib`, `seaborn`, etc).

To run the models, you will also need the `surprisal` and `openai` packages. 
See [this repo](https://github.com/aalok-sathe/surprisal/tree/main) for details and installation instructions for the `surprisal` package.

To use the OpenAI API, you will need your keys saved in a file called `openai_key.txt`
(which is ignored by `git`). The file should be a simple text file with two lines:
the first line contains your secret API key, and the second line contains your organization key.

**NOTE:** The text-davinci-* models were deprecated on January 4, 2024 (see this [announcement](https://platform.openai.com/docs/deprecations/2023-07-06-gpt-and-embeddings) from OpenAI). 

## Stimuli

The stimuli we used to evaluate the models are in the [stimuli](stimuli) folder.

There are two files:
1. `stimuli.csv` was extracted directly from
DGL's original data (see [data/dgl_original](data/dgl_original)).
2. `stimuli_minimal_pairs.csv` was made by manually annotating each of
DGL's original sentences with a minimally differing counterpart. 
Sentences in minimal pairs are matched for length and, whenever possible, lexical content.

## Running models

The models are evaluated using the script [evaluate_model.py](evaluate_model.py).
All model outputs are in the [data/model_outputs](data/model_outputs) folder. 

**NOTE:** While the results reported by DGL and in our response only use models from the
OpenAI API, the evaluation scripts are configured to also work with open-source
models on Huggingface. We hope that this encourages further exploration of these
materials in a way that is accessible and reproducible. 
See more details in the sections below.

### Minimal pair analysis (surprisals)

To obtain minimal pair results, run the following command:
```bash
python evaluate_model.py \
    -i stimuli/stimuli_minimal_pairs.csv \
    -o data/model_outputs \
    --model <MODEL> \
    --model_type <MODEL_TYPE> \
    --eval_mode minimal_pairs
```

The important variables are explained below:
- `<MODEL>` is the name of the model you would like to evaluate, which should
match either its name in the OpenAI API (e.g., `text-davinci-003`) or
the Huggingface `transformers` library (e.g., `gpt2`).
- `<MODEL_TYPE>` should be `openai` for OpenAI models, and `hf` for Huggingface models.

The command above will save data to the folder [data/model_outputs/minimal_pairs](data/model_outputs/minimal_pairs). Within this folder, there are
two subfolders: `sentence_surprisals`, which contains summed surprisals for each
sentence, and `token_surprisals`, which contains token-by-token surprisals for each
sentence.

### Prompting analysis

To obtain prompting results, run the following command:
```bash
python evaluate_model.py \
    -i stimuli/stimuli.csv \
    -o data/model_outputs \
    --model <MODEL> \
    --model_type <MODEL_TYPE> \
    --eval_mode prompting
```

The `<MODEL>` and `<MODEL_TYPE>` variables are the same as above.

**NOTE:** The code above was used to generate results from `text-davinci-002` and
`text-davinci-003`. The prompting results from GPT-3.5 Turbo and GPT-4 were obtained
using a slightly different pipeline, and thus saved in a different format
([data/model_outputs/prompting/gpt-3.5-turbo_gpt-4.csv](data/model_outputs/prompting/gpt-3.5-turbo_gpt-4.csv)).

## Analysis

All code for reproducing our figures can be found in the notebook at [notebooks/main.ipynb](notebooks/main.ipynb).
The figures are saved in PDF format to the [figures](figures) folder.
