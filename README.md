# DLoc code

This repository contains the PyTorch implementation of DLoc from [Deep Learning based Wireless Localization for Indoor Navigation](https://dl.acm.org/doi/pdf/10.1145/3372224.3380894). 

The datasets (features) required to run these codes can be downloaded from the [WILD](https://wcsng.ucsd.edu/wild/) website. You can also download the raw-channels from the [WILD](https://wcsng.ucsd.edu/wild/) webpage to run your own algorithms on them.

## DLoc Codes

To install requirements:

```setup
pip install -r requirements.txt
```


## Training and Evlautaion

To train the model(s) in the paper and evaluate them, run this command:

```train_test
python train_and_test.py
```

The file automatically imports the parameters from [params.py](params.py).

The parameters and their descriptions can be found in the comments of the example implementaion of the [params.py](params.py) file.

To recreate the results from the [paper](https://dl.acm.org/doi/pdf/10.1145/3372224.3380894) refer to the [README](./params_storage/README.md) of the **params_storage** folder.
