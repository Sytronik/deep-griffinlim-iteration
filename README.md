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

Setting `out_all_block` to `True` makes the forward method returns all outputs of `repeat` iteraions. If the output of the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;i" title="i" /></a>-th iteration is <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;o_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;o_i" title="o_i" /></a>, the loss function is defined as

<a href="https://www.codecogs.com/eqnedit.php?latex=L&space;=&space;\frac{\sum_{i=0}^{\mathrm{repeat}&space;-&space;1}\frac{1}{\mathrm{repeat}-&space;i}&space;\mathrm{L1Loss}(o_i,&space;y)}{&space;\sum_{i=0}^{\mathrm{repeat}-&space;1}&space;\frac{1}{\mathrm{repeat}-&space;i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L&space;=&space;\frac{\sum_{i=0}^{\mathrm{repeat}&space;-&space;1}\frac{1}{\mathrm{repeat}-&space;i}&space;\mathrm{L1Loss}(o_i,&space;y)}{&space;\sum_{i=0}^{\mathrm{repeat}-&space;1}&space;\frac{1}{\mathrm{repeat}-&space;i}}" title="L = \frac{\sum_{i=0}^{\mathrm{repeat} - 1}\frac{1}{\mathrm{repeat}- i} \mathrm{L1Loss}(o_i, y)}{ \sum_{i=0}^{\mathrm{repeat}- 1} \frac{1}{\mathrm{repeat}- i}}" /></a>.

## Inverse Short-time Fourier Transform (iSTFT)

The iSTFT implementation using PyTorch is in `model/istft.py`. 

The function signature convention is the same as `torch.stft`. The implementation is based on `librosa.istft`.

There is a test code under `if __name__ == '__main__'` to prove the result of this implementation is the same as that of `librosa.istft`.

The file `istft.py` doesn't have any dependency on the other files in this repository, and only depends on PyTorch.

## Requirements

- python >= 3.7 (because of `dataclass`)
- MALTAB engine for Python (because of the PESQ, STOI calculation)
- PyTorch >= 1.2 (because of the tensorboard support)
- tensorboard
- numpy
- scipy
- tqdm
- librosa
