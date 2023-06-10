
# Stable Clinical Risk Prediction against Distribution Shift in Electronic Health Records

## Introduction
This repository contains source code for paper "Stable Clinical Risk Prediction against Distribution Shift in Electronic Health Records".

The availability of large-scale electronic health record (EHR) datasets has led to the development of artificial intelligence (AI) methods for clinical risk prediction that help improve patient care.
However, existing studies have shown that AI models suffer from severe performance decay after several years of deployment, which might be caused by various temporal dataset shifts (e.g., the distribution of International Classification of Diseases (ICD) code changes when transiting ICD version from ICD-9 to ICD-10). When the dataset shift occurs, we have access to large-scale pre-shift data and small-scale post-shift data that are not enough to train new models in the post-shift environment.
In this study, we propose a new method to address the issue for clinical risk prediction. We re-weight patients collected from the pre-shift environment to mitigate the distribution shift between pre- and post-shift environments. Moreover, we adopt a Kullback–Leibler (KL) divergence loss to force the models to learn similar patient representations in pre- and post-shift environments.  Our experimental results indicate that our model trained on large-scale pre-shift data outperforms the baselines trained on small-scale post-shift data by about 17\% and improves existing models trained with pre-shift data by more than 3.5\% with our methods. 



## Overview
![figure1](https://user-images.githubusercontent.com/39074545/228349521-065e2897-2720-4d30-b9af-ba71f672afdc.png)


Figure 1: Illustrations of sample reweighting, clinical risk prediction, and the proposed method. (a) Diagram of clinical risk
prediction. (b) Changes in the distribution of medical codes after sample reweighting to mitigate the distribution shift. (c)
Architecture of the proposed method for sample reweighting.

## Installation

Our model depends on Numpy, scikit-learn, PyTorch (CUDA toolkit if use GPU), and torchmetrics. You must have them installed before using our model.



## Preprocessing data

### Dataset
Please be informed that the dataset utilized in this study is derived from MarketScan claims data. To obtain access to the data, interested parties are advised to contact IBM through the following link: [Insert link for data access].

### Input data demo
For your convenience, a demo version of the input data can be found in the data folder. It includes the data structures and a synthetic demonstration of the inputs. Prior to executing the preprocessing codes, please ensure that the format of your input data matches the format provided in the input demo. 

The detailed descriptions of each variable in the dataset can be found in the README.md in the data folder. Please refer to the README.md for comprehensive explanations of the dataset variables. 

### Parameters


### Preprocessing dataset

```python 
python preprocess/run_preprocessing.py --input_dir 'data' --pred_windows [90, 180, 360] --min_visits 10
```
>
* `input_dir`: path to datset.
* `pred_windows`: a list of prediction windows.
* `min_visits`: minimum number of visits for each patient.

## Training and test
### Python command
```python 
python train.py --config 'json/Dipole.json' --time 360 --day_dim 100 --rnn_hidden 200 --steps 500 --weight_decay 0.001 --step_lr 0.001 --target 'hf' --version 'weight' --dist_weight 1e+7 --kl_weight 1e+4 --kl_dim 64
```

### Parameters
Hyper-parameters are set in config/*.json
>
* `type`: the name of the baseline. Provided baselines are from {"Concare", "Dipole", "GRU", "LSTM", "Retain", "Stagenet"}.
* `early_stop`: the number of epochs for early stopping
* `monitor`: the criterian for early stopping. The first word is 'min' or 'max', the second one is metric.
* `metrics`: metrics to print out. It is a list format, and provided metrics are from {"accuracy", "roc_auc", "f1", "confusion"}.
* `valid_ratio`: the ratio of validation set


Hyper-parameters are set in train.py
>
* `config`: json file to use.
* `version`: from {"basic", "weight"}. "basic" and "weight" are to run the baseline and our model, respectively.
* `day_dim`: the dimension of the embedding layer. It works for {"Dipole", "GRU", "LSTM"}
* `rnn_hidden`: the number of hidden features in recurrent layers. It works for {"Dipole", "GRU", "LSTM", "Retain"}
* `weight_decay`: weight decay when training the predictive model (baseline)
* `steps`: the number of epochs to learn sample weights
* `step_lr`: learning rate to learn sample weights
* `kl_dim`: the number of hidden features in Autoencoder
* `kl_weight and dist_weight`: weights to control KL and MSE losses, respectively
