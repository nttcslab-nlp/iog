# Input-to-Output Gate for Language Models


This repository contains the Input-to-Output gate proposed in [Input-to-Output Gate to Improve RNN Language Models](https://arxiv.org/abs/1709.08907) published in IJCNLP 2017.
In addition, this repository also contains various RNN language models including [LSTM with dropout](https://arxiv.org/abs/1409.2329), [variational LSTM](https://arxiv.org/abs/1512.05287), and [Recurrent Highway Network](https://arxiv.org/abs/1607.03474) for the base RNN language models.


If you use this code or our results in your research, please cite:

```
@inproceedings{iog4lm,
  title={{Input-to-Output Gate to Improve RNN Language Models}},
  author={Takase, Sho and Suzuki, Jun and Nagata, Masaaki},
  booktitle={Proceedings of the 8th International Joint Conference on Natural Language Processing (IJCNLP 2017)},
  year={2017}
}
```

## Software Requirements

* Python 2
* Chainer 1.X (recommend version 1.24)

We note that it is impossible to run these codes on Python 3 or chainer 2.X.


## How to use

* Run `preparedata.sh` to obtain the preprocessed Penn Treebank dataset and construct .npz files.
* Train the base language model
    * E.g., to obtain [LSTM language model with dropout](https://arxiv.org/abs/1409.2329), run

```
python learnLSTMLM.py -g 0 --train data/ptb/ptb.train.npz --valid data/ptb/ptb.valid.npz -v data/ptb/ptb.train.vocab
   --valid_with_batch --output mediumLSTMLM
```

* Train Input-to-Output gate for the above base language model
    * E.g., run the following command to train Input-to-Output gate for the lstm language model in the above example.

```
python learnIOG.py -g 0 --train data/ptb/ptb.train.npz --valid data/ptb/ptb.valid.npz -s mediumLSTMLM.setting
   -m mediumLSTMLM.epoch39_fin.bin --output iog4mediumLSTMLM
```

* Run `testLM.py` for computing perplexity of the base language model
    * E.g., run `python testLM.py -g 0 -t data/ptb/ptb.test.npz -s mediumLSTMLM.setting -m mediumLSTMLM.epoch39_fin.bin` for the lstm language model in the above example.
* Run `testIOG.py` for computing perplexity of the base language model with Input-to-Output gate.
    * E.g., run `python testIOG.py -g 0 -t data/ptb/ptb.test.npz -s iog4mediumLSTMLM.setting -m iog4mediumLSTMLM.bin` for the Input-to-Output gate in the above example.
* If you want to use the [cache mecanism](https://arxiv.org/abs/1612.04426), specify `--cachesize N` in running `testLM.py` or `testIOG.py`


