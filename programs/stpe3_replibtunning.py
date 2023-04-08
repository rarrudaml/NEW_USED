###########################
######
###### Replib for title field - train
#####
###########################
###### Packages
from replib.descriptors.metaprod2vec import MetaProd2Vec
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
###########################


def step03_replib_train(df_train):
        
    # RF
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = make_pipeline(
        MetaProd2Vec(pre_fitted_version="MLA_feb21_dim100")
        ,RandomForestClassifier(**params)
    )
    
    targets = df_train['condition']
    titles = df_train["title"].fillna('XXXX').tolist()
    items = [{"title": title} for title in titles]
    
    return model.fit(items, targets)  

###########################
######
###### Replib for title field - predict
#####
###########################

def step04_replib_predict(df_train,df_test,model_replib):
    
    titles_train = df_train["title"].fillna('XXXX').tolist()
    items_train = [{"title": title} for title in titles_train]
    df_train['trat_rep_title'] = model_replib.predict_proba(items_train)[:, 0]
    
    titles_test = df_test["title"].fillna('XXXX').tolist()
    items_test = [{"title": title} for title in titles_test]
    df_test['trat_rep_title'] = model_replib.predict_proba(items_test)[:, 0]
    
    return df_train, df_test