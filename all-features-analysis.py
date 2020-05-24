'''
THIS FILE HAS NEVER BEEN EXECUTED AS IT IS.
IT CONTAINS THE FUNCTIONS AND THE PROCESS OF ANALYZING THE TEXT FOR THIS PROJECT THAT WAS EXECUTED
IN JUPYTER NOTEBOOKS. FOR ANY QUESTION CONTACT WITH @myrthings
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from copy import deepcopy
import string
import seaborn as sns
import emoji
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,auc,roc_curve,cohen_kappa_score,precision_score,recall_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import MultinomialNB

from matplotlib import rc

# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':12})

# Set the font used for MathJax - more on this later
rc('mathtext',**{'default':'regular'})

np.random.seed(101)

def oversample(df,factor=20,perc=0.1):
    '''
    Given a dataset this function oversample it by a factor.
    It returns the train and test set.
    '''
    df['y'].value_counts()

    df0=shuffle(df[df['y']==0]).reset_index(drop=True)
    df1=shuffle(df[df['y']==1]).reset_index(drop=True)

    len1_test=round(len(df1)*perc)
    
    df1_test=pd.concat([df1[:len1_test]]*factor,axis=0)
    df1_train=pd.concat([df1[len1_test:]]*factor,axis=0)
    
    len0_test=len(df1_test)
    len0_train=len(df1_train)
    
    df0_test=df0[:len0_test]
    df0_train=df0[len0_test:len0_test+len0_train]
    
    train=shuffle(pd.concat([df0_train,df1_train],axis=0))
    test=shuffle(pd.concat([df0_test,df1_test],axis=0))
    
    return train, test

def apply_models(models,X_train,y_train,X_test,y_test):
    '''
    Given a dictionary of ml models, a train and test set, this function applies and
    evaluates all those models.
    It returns a df with the metrics and the feature importance of the models
    '''
    df_metrics=pd.DataFrame([])
    feature_imp=pd.DataFrame([])

    for model_name in models.keys():
        print(model_name)
        pos=0
        metrics={}
        model=models[model_name]
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        try:
            probs=model.predict_proba(X_test)
            pos=1
        except:
            pos=0
        conf=confusion_matrix(y_test,y_pred)

        metrics['accuracy']=accuracy_score(y_test,y_pred)
        metrics['cohen_kappa']=cohen_kappa_score(y_test,y_pred)
        if pos==1:
            metrics['roc_auc']=roc_auc_score(y_test,[item for _,item in probs])
        else:
            metrics['roc_auc']=np.nan

        df_metrics=pd.concat([df_metrics,pd.DataFrame(metrics,index=[model_name]).T],axis=1)
        try:
            feature_imp=pd.concat([feature_imp,pd.DataFrame({'variable':X_train.columns,'imp_'+model_name:model.feature_importances_}).set_index('variable')],axis=1)
        except:
            None
    return df_metrics, feature_imp


## ml models dictionary with the basic parameters
models={
    'logmodel':LogisticRegression(random_state=101),
    'decisiontree':DecisionTreeClassifier(),
    'randomforest':RandomForestClassifier(),
    'adamodel':AdaBoostClassifier(n_estimators=10,random_state=101),
    'xgbmodel':XGBClassifier(random_state=101),
    'gradientmodel':GradientBoostingClassifier(n_estimators=10,random_state=101),
    'lightmodel':LGBMClassifier(random_state=101),
    'catmodel':CatBoostClassifier(silent=True,random_state=101),
    'naivebayes':MultinomialNB()
}


if __name__ == "__main__":
    df=pd.read_csv('complete_dataset.csv').drop('Unnamed: 0',axis=1)
    df=df.drop(['_id','RS','RS_cat','tuit_topic_contrib','tuit_tsne_x','tuit_tsne_y','clean3','created_hour','franja',
           'user_handler','user_createdyear','user_age','influence','exclusivity','tuitero','user_topic_contrib',
           'user_tsne_x','user_tsne_y'],axis=1)

    df['y']=df['y'].astype(int)

    ## convert to numbers
    lang_cat=dict(zip(df['user_descrip_lang'].unique(),range(-1,3,1)))
    df['user_descrip_lang']=df['user_descrip_lang'].apply(lambda x: lang_cat[x])
    df['langdetect']=df['langdetect'].apply(lambda x: lang_cat[x])
    tuit_cat=dict(zip(df['tuit_topic'].unique(),range(df['tuit_topic'].nunique())))
    descrip_cat=dict(zip(df['user_topic'].unique(),range(df['user_topic'].nunique())))
    df['tuit_topic']=df['tuit_topic'].apply(lambda x: tuit_cat[x])
    df['user_topic']=df['user_topic'].apply(lambda x: descrip_cat[x])

    # naive bayes doesn't support negative numbers
    for col in df.columns:
        df[col]=df[col].apply(lambda x: 10 if x<0 else x)


    ## oversample and divide in train and test sets
    df_train,df_test=oversample(df)

    y_test=df_test['y']
    X_test=df_test.drop('y',axis=1)

    y_train=df_train['y']
    X_train=df_train.drop('y',axis=1)


    ## apply the models
    df_metric,df_imp=apply_models(models,X_train,y_train,X_test,y_test)




    ## plot the results
    fig,ax=plt.subplots(1,3,figsize=(20,5))

    for i,model_name in enumerate(['lightmodel','xgbmodel','catmodel']):

        model=models[model_name]
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        y_probs=model.predict_proba(X_test)

        preds = y_probs[:,1]
        fpr, tpr, threshold = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)

        ax[i].set_title('ROC {}'.format(model_name))
        ax[i].plot(fpr, tpr, color='lightseagreen',label = 'AUC = %0.4f' % roc_auc)
        ax[i].legend(loc = 'lower right')
        ax[i].plot([0, 1], [0, 1],color='coral')
        ax[i].set_xlim([0, 1])
        ax[i].set_ylim([0, 1])
        ax[i].set_ylabel('True Positive Rate')
        ax[i].set_xlabel('False Positive Rate')


        
    plt.show()






