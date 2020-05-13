# BART version of closed-book QA

This is a BART version of sequence-to-sequence model for open-domain QA in a closed-book setup, based on [PyTorch](https://pytorch.org/) and [Huggingface's Transformers](https://github.com/huggingface/transformers).

The model is a sequence-to-sequence model that takes a question as an input and outputs the answer, without reading any external resource (e.g. passages).
Please refer to [Roberts et al., 2020, How Much Knowledge Can You Pack Into the Parameters of a Language Model?](https://arxiv.org/abs/2002.08910) to learn more about closed-book QA setup and the original model based on T5. Their code and model checkpoints are available [here](https://github.com/google-research/google-research/tree/master/t5_closed_book_qa).

The model is based on BART-large. Please refer to [Lewis et al., ACL 2020, BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) to learn more about BART.

We experiment with Natural Questions open-domain data (NQ-open), but the code should work on any QA data with question-answer pairs.


## Requirement

This code is tested on Python 3.6.9.

Install PyTorch and Transformers:
```
pip install torch==1.1.0
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
```

Download NQ-open data:
```
chmod +x download_data.sh; ./download_data.sh
```

## Training

```
python cli.py --do_train --output_dir out/nq-bart-closed-qa \
        --train_file data/nqopen-train.json \
        --predict_file data/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --append_another_bos
```

The script will save the log and the best checkpoint inside `out/nq-bart-closed-qa`.


Other useful commands (please refer to `cli.py` for the full list):
- `eval_period`: interval to evaluate on the dev data
- `verbose`: print a progress bar
- `debug`: train and evaluate on a subset of the dev data for debugging purposes

You can use `train_batch_size` and `predict_batch_size` depending on the gpu availability. With one 16GB gpu, you can use `train_batch_size=64, predict_batch_size=64`.
Our model that we reports the result below was trained with `train_batch_size=1024, predict_batch_size 256` using eight 32GB gpus. Training took roughly 34 hours.

Note:
- This script saves the pre-tokenized data in `data/` once question-answer pairs are tokenized for the first time.
- The model gives the best result when prepending extra BOS token (`--append_another_bos`).
- Inference on multi-gpus is not working for now; we will update the code once it is fixed.

## Inference

```
python cli.py --do_predict --output_dir out/nq-bart-closed-qa \
        --predict_file data/nqopen-dev.json \
        --predict_batch_size ${test_bs} \
        --append_another_bos --prefix dev_
python cli.py --do_predict --output_dir out/nq-bart-closed-qa \
        --predict_file data/nqopen-test.json \
        --predict_batch_size ${test_bs} \
        --append_another_bos --prefix test_
```

It will save the prediction file as `out/nq-bart-closed-qa/{dev|test}_predictions.json`.

## Result

The final Exact Match score we get is 25.05 on the dev data and 24.10 on the test data.

We made the best model checkpoint and the predictions on the dev/test data available.

- [Best checkpoint + Dev/Test prediction (1.8G)][1]
- [Dev/test predictions only (228K)][2]

Note that T5-based model gets 27.0, 29.8, 32.1 and 34.5 on the test set with Base, Large, 3B and 11B, respectively, based on [the original paper](https://arxiv.org/pdf/2002.08910.pdf). Several factors could lead to the performance gaps: (i) T5 has a larger number of parameters and trained on a larger set of data and (ii) the original paper includes the dev data for training, whereas our codebase only trains the model on the train data and uses the dev data for choosing the best checkpoint.
We also did not perform any hyperparamter tuning, as our goal is to provide the basic codebase rather than to achieve the best possible performance; we leave it for the future work.

Note: that the original paper includes ablations that exclude supervised data for T5 pretraining, and reports comparable (or better) numbers: see Appendix C of [the original paper](https://arxiv.org/pdf/2002.08910.pdf) for the details!

## Contact

Please email [Sewon Min](https://shmsw25.github.io) or write a Github issue for any question.


[1]: http://nlp.cs.washington.edu/ambigqa/models/nq-bart-closed-qa/nq-bart-closed-qa.zip
[2]: http://nlp.cs.washington.edu/ambigqa/models/nq-bart-closed-qa/predictions.zip


