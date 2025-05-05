# Diff-SE: A Diffusion-Augmented Contrastive Learning Framework for Super-Enhancer Prediction

## 1. Generate manual features

Install the repdna library

- *pip install RepDNA* 

Use the installed library to generate manual features and save them in *dataset/cell_name/* directory

## 2. File descriptions

classification_experiment：Article experiments section code

dataset：The path where the dataset is stored

diff：Diffusion model code

generate_experiment：Diffusion model experiment code

Second：Classification model training code

## 3. Environment setup

We recommend you to build a python virtual environment with Anaconda. 


#### 3.1 Create and activate a new virtual environment

```
conda create -n DiffSE python=3.9
conda activate DiffSE
```



#### 3.2 Install the package and other requirements

```
python -m pip install -r requirements.txt
```
If you want to download the pytorch environment separately, execute this command
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
