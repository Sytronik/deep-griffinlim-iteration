# deep-griffinlim-iteration
PyTorch implementation for [Deep Griffin-Lim Iteration paper](https://arxiv.org/abs/1903.03971)

## Usage

All configurations are in `hparams.py`.

`create.py` saves STFT of of the speech data.

To train DNN, use `python main.py --train`.

To test DNN, use `python main.py --test`.

## Model Change

Unlike the paper, the DNN model contains BatchNorm layers and Conv layers with larger (7x7) kernel size.

If the hyperparameter `depth` is greater than 1, the model performs the deep-griffinlim iteration `depth` times, and use separate DNN for each iteration.

If the `repeat` argument of the forward method is greater than 1, the model repeats the `depth` iterations `repeat` times by reusing the DNN models.

(To use the same single DNN for all iterations, set `depth=1` and `repeat>1`.)

Setting `out_all_block` to `True` makes the forward method returns all outputs of all `depth * repeat` iteraions. If the output of the `i`-th iteration is `o_i`, the loss function is defined as

$$
    L = \sum_{i=0}^{depth \cdot repeat - 1}\frac{1}{repeat - i} L1Loss(o_i, y) / \sum_{i=0}^{depth \cdot repeat - 1} \frac{1}{repeat - i}.
$$
