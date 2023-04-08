# NEW vs USED Challenge:

## Objective:

In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the marketplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function `build_dataset` to read that dataset in `new_or_used.py`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. Additionally, you will have to choose an appropriate secondary metric and also elaborate an argument on why that metric was chosen.


## Pre-reqs:

Enable some virtual env with required packages:
`virtualenv --python=/usr/bin/python3 venv`
`source venv/bin/activate`

## Modules:

`model_new_vs_used_v1.py`: Contain __main__ function. This program load other modules necessary
`step0_read.py`: import datasets (train, test)
`step1_preproc`: feature engineering module. Mainly focused in obtain features from dict objects
`step2_warranty.py`: Build an embedding of words to waaranty filed
`stpe3_replibtunning.py`: Train an RF using Replib transform of item field
`step4_finaldataset.py`: Remove unsed filds
`step5_modeltrainning.py`: Train final model

## Steps to run program:

All modules, including `model_new_vs_used_v1.py` should be in the same directory as the input dataset.
In the terminal enter in the directory and exceute: `python model_new_vs_used_v1.py`

