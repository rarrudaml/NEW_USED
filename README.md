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

Avaible in `/programs` folder

## Steps to run program:

All modules, including `model_new_vs_used_v1.py` should be in the same directory as the input dataset.
In the terminal enter in the directory and exceute: `python model_new_vs_used_v1.py`


## Logs:

Loading dataset...
Pre-processing TRAIN dataset running...
Pre-processing TEST dataset running...
Pre-processing: DONE!
Text Processing - Warranty: TRAIN dataset running...
Text Processing - Warranty: TEST dataset running...
Text Processing - Warranty: DONE!
Replib: TRAINNING running...
Replib training: DONE!
Replib: PREDICT running...
Replib predict: DONE!
DATASET cleaning running ...
DATASET cleaning: DONE!
TRAINNING ML model...
Model TRAINNING: DONE!
predicting TEST SET running ...
predicting TEST SET: DONE!

## Result:

****************************************************************************
**************************    PROCESS FINISHED !     ***********************
****************************************************************************
Accuracy in TEST SET =  0.8822


## Secundary metric:

The dataset is balanced: 53,7% in train set. So we don't have motivation to be worried about rare effects of Accuracy.

Accuracy is a commonly used metric in machine learning, but it requires a cutoff value to be defined, which can be time-consuming during the development stage. The choice of the cutoff value can have a significant impact on the accuracy of the model, and in some cases, it may not be obvious which value to choose.

On the other hand, the AUC_ROC (Area Under the Receiver Operating Characteristic Curve) is a powerful metric that measures the performance of a binary classification model without the need for a cutoff value. It takes into account the sensitivity and specificity of the model across all possible cutoff values, providing a comprehensive view of the model's performance. This makes it a great metric for comparing models and selecting the best one for a given task.

AUC_ROC in TEST SET =  0.9519041831171319


