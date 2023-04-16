###########################
######
###### Train final model
#####
###########################
###### Packages
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

###########################


def step05_model_train(df_train):

    # RF
    params = {
        'bootstrap': True,
        'ccp_alpha': 0.0,
        'class_weight': None,
        'criterion': 'gini',
        'max_depth': None,
        'max_features': 'sqrt',
        'max_leaf_nodes': None,
        'max_samples': None,
        'min_impurity_decrease': 0.0,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'min_weight_fraction_leaf': 0.0,
        'n_estimators': 100,
        'n_jobs': -1,
        'oob_score': False,
        'random_state': 123,
        'verbose': 0,
        'warm_start': False,
    }

    model = make_pipeline(RandomForestClassifier(**params))

    targets = df_train['condition']
    X = df_train.drop(columns=['condition'])

    return model.fit(X, targets)
