# Source code for our paper

### Requirements

- Python 3.8.5
- torch 1.7.1
- CUDA 11.3

### Dataset

The raw data can be found at [IEMOCAP](https://sail.usc.edu/iemocap/ "IEMOCAP") and [MELD](https://github.com/SenticNet/MELD "MELD").

In our paper, we use pre-extracted features. The multimodal features are available at [here](https://www.dropbox.com/sh/4b21lympehwdg4l/AADXMURD5uCECN_pvvJpCAy9a?dl=0 "here").


### Training examples

For instance, to train on IEMOCAP:
`python -u train.py --base-model 'LSTM' --dropout 0.5 --lr 0.00009 --batch-size 16 --model_type='emoplex' --epochs=120 --multi_modal --modals='avl' --Dataset='IEMOCAP' --norm LN`
