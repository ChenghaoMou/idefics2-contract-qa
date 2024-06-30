# Fine-tuning Idefics2 on EDGAR Contract QA Dataset

See [Blog](https://sleeplessindebugging.blog/posts/20240630131948) for more details.

1. `dataset.ipynb` prepares the dataset.
2. `train.py` fine-tunes the model on the dataset.
3. `benchmark.ipynb` evaluates the model on the test dataset.

## Datasets

[chenghao/sec-material-contracts-qa-splitted](https://huggingface.co/datasets/chenghao/sec-material-contracts-qa-splitted) consists of the following data:
1. [chenghao/sec-material-contracts-qa](https://huggingface.co/datasets/chenghao/sec-material-contracts-qa)
2. [jordyvl/DUDE_subset_100val](https://huggingface.co/datasets/jordyvl/DUDE_subset_100val)

Data splits: train (80%), test (20%)

## Model

More details can be found at [idefics2-edgar](https://huggingface.co/chenghao/idefics2-edgar). The training script can be run with a single GPU (A100-80GB) with low resolution input and QLoRA training.

## References:
1. [@NielsRogge](https://github.com/NielsRogge)'s tutorial [Fine_tune_Idefics2_for_multi_page_PDF_question_answering_on_DUDE.ipynb](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Idefics2/Fine_tune_Idefics2_for_multi_page_PDF_question_answering_on_DUDE.ipynb)
2. [Idefics2](https://huggingface.co/transformers/model_doc/idefics2.html)