# CGF
This is a PyTorch implementation of the paper: CGF: A Category Guidance Based PM2.5 Sequence
Forecasting Training Framework. In order to show our modules in framework
 more clearly, we plug CGF into a simple Encoder-Decoder struture.
`@author:Haomin Yu, Jilin Hu, Xinyuan Zhou, Chengjuan Guo, Bin Yang, Qingyong Li`


## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt
- arrow==0.15.4
- bresenham==0.2.1
- dgl==0.6.1
- geopy==1.20.0
- matplotlib==3.1.1
- MetPy==0.12.1
- numpy==1.19.5
- pandas==1.1.5
- scipy==1.4.1
- torch==1.5.1+cu92
- torch_geometric==1.6.0
- torch_scatter==2.0.5
- tqdm==4.38.0

## Datasets

###Download Datasets
Download KnowAir from (https://github.com/shawnwang-tech/PM2.5-GNN)

### Preparing datasets
The data should be in a separate folder called "data" inside the project folder.
The details of preprocessing is as follows.


### Processing dataset
Select some regions for training, and add the indexes of these regions to  
`config_dict["region_index"]` in `config.py` accroding to  `city.txt`.
In our work, we focus on the regions of Hebei and Zhejiang.


## Model Training
Set `config_dict['IsTrain'] = True` in `config.py`, and then run `main.py`.

```
python main.py 

```

## Model Testing


Set `config_dict['IsTrain'] = False` in `config.py`, and then run `main.py`.

```
python main.py 

```

## Model Hyperparameters

You can adjust `config_dict['expParam'] ` in `config.py` 
according to the prediction scenario. Note that `config_dict['expParam']` includes some important hyperparameters in CGF.