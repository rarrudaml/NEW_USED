###########################
######
###### Main Module
#####
###########################
###### Packages
from step0_read import build_dataset
from step1_preproc import step01_preproc
from step2_warranty import step02_waranty_transform
from step3_replibtunning import step03_replib_train, step04_replib_predict
from step4_finaldataset import step04_clean_dataframes
from step5_modeltrainning import step05_model_train
from sklearn.metrics import accuracy_score

###########################


if __name__ == '__main__':
    print('Loading dataset...')
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()
    # Step Pre-processing
    print('Pre-processing TRAIN dataset running...')
    df_train = step01_preproc(X_train)
    print('Pre-processing TEST dataset running...')
    df_test = step01_preproc(X_test)
    print('Pre-processing: DONE!')
    # Step Text Processing - Warranty
    print('Text Processing - Warranty: TRAIN dataset running...')
    df_train = step02_waranty_transform(df_train)
    print('Text Processing - Warranty: TEST dataset running...')
    df_test = step02_waranty_transform(df_test)
    print('Text Processing - Warranty: DONE!')
    # Step Training Replib
    print('Replib: TRAINNING running...')
    model_replib = step03_replib_train(df_train)
    print('Replib training: DONE!')
    # Step Predict Replib
    print('Replib: PREDICT running...')
    df_train, df_test = step04_replib_predict(df_train, df_test, model_replib)
    print('Replib predict: DONE!')
    # Step Clean Dataset
    print('DATASET cleaning running ...')
    df_train, df_test = step04_clean_dataframes(df_train, df_test)
    print('DATASET cleaning: DONE!')
    # Train model
    print('TRAINNING ML model...')
    model_final = step05_model_train(df_train)
    print('Model TRAINNING: DONE!')
    # Predict TEST
    print('predicting TEST SET running ...')
    predictions = model_final.predict(df_test)
    print('predicting TEST SET: DONE!')
    print(
        '****************************************************************************'
    )
    print(
        '**************************    PROCESS FINISHED !     ***********************'
    )
    print(
        '****************************************************************************'
    )
    print('Accuracy in TEST SET = ', accuracy_score(y_test, predictions))
