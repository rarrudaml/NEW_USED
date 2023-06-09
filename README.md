# NEW vs USED Challenge: Solution


## Objective:


The goal of this project is to build an algorithm to predict whether an item listed in MercadoLibre's Marketplace is new or used, using machine learning.

The tasks involved in this project include data analysis, designing, processing, and modeling of a machine learning solution to predict if an item is new or used. After building the model, it is evaluated over held-out test data using the accuracy metric, with a minimum threshold of 0.86. Additionally, an appropriate secondary metric is chosen, and an argument is presented on why that metric was chosen.



## Pre-reqs:

The project requires a virtual environment with the required packages (see requirements.txt). To enable the virtual environment, run the following command:

`virtualenv --python=/usr/bin/python3 venv`

To activate the virtual environment, run the following command:

`source venv/bin/activate`




## Modules:

The program contains the following modules:

`model_new_vs_used_v1.py`: The main function that loads other necessary modules.

`step0_read.py`: Imports datasets (train, test).

`step1_preproc`: A feature engineering module that obtains features from dict objects.

`step2_warranty.py`: Builds an embedding of words to the warranty field.

`step3_replibtunning.py`: Trains an RF using the Replib transform of the item field.

`step4_finaldataset.py`: Removes unused fields.

`step5_modeltrainning.py`: Trains the final model.

All modules are available in the `/programs` folder.



## Steps to run program:


All modules, including `model_new_vs_used_v1.py`, should be in the same directory as the input dataset. To run the program, navigate to the directory and execute the following command:

`python model_new_vs_used_v1.py`


## Logs:


During the program execution, logs are displayed in the terminal. The following logs are displayed:

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


The dataset used for this project is balanced, with 53.7% new products in the train set. As a result, rare effects of accuracy are not a concern.

While accuracy is a commonly used metric in machine learning, it requires a cutoff value to be defined, which can be time-consuming during the development stage. The choice of the cutoff value can have a significant impact on the accuracy of the model, and in some cases.

On the other hand, the AUC_ROC (Area Under the Receiver Operating Characteristic Curve) is a powerful metric that measures the performance of a binary classification model without the need for a cutoff value. It takes into account the sensitivity and specificity of the model across all possible cutoff values, providing a comprehensive view of the model's performance. This makes it a great metric for comparing models and selecting the best one for a given task in balanced dataset scenario.
