# Summarization

Multi-Document Guided Summarization

Repository for the project component of the LING 575 - Summarization class.

Student team:

- Anna Batra
- Sam Briggs
- Junyin Chen
- Hilly Steinmetz

## Getting started

### Condor or Locally

- Install [conda](https://docs.anaconda.com/anaconda/install/index.html)
- `conda env create -f environment.yaml` to initialize conda environment
- Put the downloaded the `pytorch_model.bin` model file from [Hugging Face](https://huggingface.co/junyinc/LING-575-WI-SUM/tree/main) in `outputs/`

### Evaluation
- As patas does not have the necessary package installed to run ROUGE-1.5.5pl, you will have to run the tool manually on your local device.
- We have provided the config file 'rouge_run_D3.xml' in src, please change the [REPLACE PEER ROOT] and [REPLACE MODEL ROOT] with the Peer Root and Model Root on your local machine.
