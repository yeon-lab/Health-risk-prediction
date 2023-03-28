# Stable Clinical Risk Prediction against Distribution Shift in Electronic Health Records

## Introduction
This repository contains source code for paper "Stable Clinical Risk Prediction against Distribution Shift in Electronic Health Records".

The availability of large-scale electronic health record (EHR) datasets has led to the development of artificial intelligence (AI) methods for clinical risk prediction that help improve patient care.
However, existing studies have shown that AI models suffer from severe performance decay after several years of deployment, which might be caused by various temporal dataset shifts (e.g., the distribution of International Classification of Diseases (ICD) code changes when transiting ICD version from ICD-9 to ICD-10). When the dataset shift occurs, we have access to large-scale pre-shift data and small-scale post-shift data that are not enough to train new models in the post-shift environment.
In this study, we propose a new method to address the issue for clinical risk prediction. We re-weight patients collected from the pre-shift environment to mitigate the distribution shift between pre- and post-shift environments. Moreover, we adopt a Kullback–Leibler (KL) divergence loss to force the models to learn similar patient representations in pre- and post-shift environments.  Our experimental results indicate that our model trained on large-scale pre-shift data outperforms the baselines trained on small-scale post-shift data by about 17\% and improves existing models trained with pre-shift data by more than 3.5\% with our methods. 



## DREAM
![Figure2_org](https://user-images.githubusercontent.com/39074545/208546254-11d0bcb9-a573-43ab-9ef6-6039760112bc.png)


Figure 1: Overall architecture of **DREAM**

## Installation

Our model depends on Numpy, scikit-learn, PyTorch (CUDA toolkit if use GPU), and pytorch_metric_learning. You must have them installed before using our model.



## Usage

### Training and test
```python 
python train.py --fold_id=0 --np_data_dir "data_npz/edf_20_fpzcz" --config "config.json"
```

### Hyper-parameters
Hyper-parameters are set in config.json
>
* `seq_len`: Length of input sequence for classification network
* `n_layers`: the number of encoder layers in Transformer
* `num_folds`: the number of folds for k-fold cross-validation
* `early_stop`: the number of epochs for early stopping
* `monitor`: the criterian for early stopping. The first word is 'min' or 'max', the second one is metric.
* `const_weight`: a weight to control constrastive loss
* `dim_feedforward`: the dimension of the feedforward network model in Transformer encoder layer
* `beta_d and beta_y`: weights to control KL losses for subject and class, respectively
* `zd_dim and zy_dim`: output dimensions of subject and class encoders, respectively
* `aux_loss_d and aux_loss_y`: weights to control auxiliary losses for subject and class, respectively


