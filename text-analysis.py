'''
THIS FILE HAS NEVER BEEN EXECUTED AS IT IS.
IT CONTAINS THE FUNCTIONS AND THE PROCESS OF ANALYZING THE TEXT FOR THIS PROJECT THAT WAS EXECUTED
IN JUPYTER NOTEBOOKS. FOR ANY QUESTION CONTACT WITH @myrthings
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import string
import emoji

from sklearn.utils import shuffle
from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,auc,roc_curve,cohen_kappa_score,precision_score,recall_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, SimpleRNN, LSTM
from keras import layers

from matplotlib import rc

# Set the global font
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':12})
rc('mathtext',**{'default':'regular'})

np.random.seed(101)

def oversample_numpy(X1,X0,y1,y0,factor=20,perc=0.1):
    '''
    This function is to divide the sample into training and test set using an oversampling method of factor "factor"
    '''

    len1_test=round(len(y1)*perc)
    
    y1_test=pd.concat([y1[:len1_test]]*factor,axis=0)
    y1_train=pd.concat([y1[len1_test:]]*factor,axis=0)
    
    X1_test=np.concatenate([X1[:len1_test]]*factor)
    X1_train=np.concatenate([X1[len1_test:]]*factor)
    
    len0_test=len(y1_test)
    len0_train=len(y1_train)
    
    y0_test=y0[:len0_test]
    y0_train=y0[len0_test:len0_test+len0_train]
    
    X0_test=X0[:len0_test]
    X0_train=X0[len0_test:len0_test+len0_train]
    
    X_train=np.concatenate([X1_train,X0_train])
    X_test=np.concatenate([X1_test,X0_test])
    
    y_train=np.array(pd.concat([y1_train,y0_train],axis=0).to_list())
    y_test=np.array(pd.concat([y1_test,y0_test],axis=0).to_list())
    
    X_train, y_train = shuffle(X_train, y_train, random_state=101)
    X_test, y_test = shuffle(X_test, y_test, random_state=101)
    
    return X_train,y_train,X_test,y_test

## This dictionary is to start all models used with basic estimators
ml_models={
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

def apply_ml_models_text(models,X_train,y_train,X_test,y_test):
    '''
    This function applies all models in models dictionary to the train and test sets given.
    It returns a df with the metrics, and two dictionaries with the predictions and the probabilities of each model
    '''
    df_metrics=pd.DataFrame([])
    y_preds={}
    y_probs={}
    
    for model_name in models.keys():
        print(model_name)
        pos=0
        metrics={}
        model=models[model_name]
        try:
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            y_preds[model_name]=y_pred
            try:
                probs=model.predict_proba(X_test)
                y_probs[model_name]=probs
                pos=1
            except:
                pos=0
                y_probs[model_name]=[]

            metrics['accuracy']=accuracy_score(y_test,y_pred)
            metrics['cohen_kappa']=cohen_kappa_score(y_test,y_pred)
            if pos==1:
                metrics['roc_auc']=roc_auc_score(y_test,[item for _,item in probs])
            else:
                metrics['roc_auc']=np.nan

            df_metrics=pd.concat([df_metrics,pd.DataFrame(metrics,index=[model_name]).T],axis=1)
        except Exception as e:
            print(model_name,':',e)
        
    return df_metrics,y_preds,y_probs


def rocs_top3(models_names,y_test,y_probs,name):
    '''
    This function plots 3 roc curves given 3 names (in model_names), the test set and the probabilities dictionary.
    Also the name is given to plot the titles and saves the fig.
    '''

    fig,ax=plt.subplots(1,3,figsize=(20,5))


    for i,model_name in enumerate(models_names):

        preds = y_probs[model_name][:,1]
        fpr, tpr, threshold = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)

        ax[i].set_title('ROC {} {}'.format(name,model_name))
        ax[i].plot(fpr, tpr, color='lightseagreen',label = 'AUC = %0.4f' % roc_auc)
        ax[i].legend(loc = 'lower right')
        ax[i].plot([0, 1], [0, 1],color='coral')
        ax[i].set_xlim([0, 1])
        ax[i].set_ylim([0, 1])
        ax[i].set_ylabel('True Positive Rate')
        ax[i].set_xlabel('False Positive Rate')



    plt.savefig('roc {} {}.png'.format(name,','.join(models_names)))
    plt.show()


####### BELOW ARE ALL THE DL MODELS USED

def basic_nn(input_shape):
    model=Sequential()
    model.add(Dense(64, activation='sigmoid',input_shape=input_shape))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1,activation='sigmoid',))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
    model.summary()
    
    return model

def basic_rnn(input_shape):

    model=Sequential()
    model.add(SimpleRNN(32,activation='sigmoid',input_shape=input_shape))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    
    return model


def basic_lstm(input_shape):
    model=Sequential()
    model.add(LSTM(32,activation='sigmoid',input_shape=input_shape))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    
    return model

    
def basic_conv(input_shape):
    model = Sequential()
    model.add(layers.Conv1D(32, 7, activation='relu',input_shape=input_shape))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    
    return model
    
    
dl_models={
    'perceptron':basic_nn,
    'rnn':basic_rnn,
    'lstm':basic_lstm,
    'convolutional_networks':basic_conv
}



def apply_dl_models(models,X_train,y_train,X_test,y_test,ep=10):
    '''
    This function applies all the models given by models dict to the train and test set
    with a number of epochs (ep) given.
    It returns a dictionary with the models trained, a dictionary with models history and
    a dictionary with the probabilities.
    '''
    trained_models={}
    history_dic={}
    preds={}
    for model_name in models.keys():
        print(model_name)
        
        if model_name=='perceptron':
            trained_models[model_name]=models[model_name]((X_train.shape[1],))
            history_dic[model_name]=trained_models[model_name].fit(X_train,y_train,
                 epochs=ep,
                 batch_size=32,
                 validation_split=0.1)
            preds[model_name]=trained_models[model_name].predict_proba(X_test)
        
        else:
            X_train_mej=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
            X_test_mej=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
            
            trained_models[model_name]=models[model_name]((X_train_mej.shape[1],1))
            history_dic[model_name]=trained_models[model_name].fit(X_train_mej,y_train,
                 epochs=ep,
                 batch_size=32,
                 validation_split=0.1)
            preds[model_name]=trained_models[model_name].predict_proba(X_test_mej)
    
    return trained_models,history_dic,preds



def plot_loss_acc_roc(model_history,y_test,y_probs,name):
    '''
    This function plots loss, accuracy and the roc curve for a given model (name).
    It needs the model history, the test and the probabilities.
    '''

    fig,ax=plt.subplots(1,3,figsize=(20,5))
    
    history_dict = model_history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    
    ax[0].plot(epochs, loss_values, marker='o',color='lightseagreen', linewidth=0,label='Training loss')
    ax[0].plot(epochs, val_loss_values, color='lightseagreen', label='Validation loss')
    ax[0].set_title('{} training and validation loss'.format(name))
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    
    ax[1].plot(epochs, acc_values, marker='o',color='lightseagreen', linewidth=0, label='Training accuracy')
    ax[1].plot(epochs, val_acc_values, color='lightseagreen', label='Validation accuracy')
    ax[1].set_title('{} training and validation accuracy'.format(name))
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    
    


    fpr, tpr, threshold = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    ax[2].set_title('{} ROC curve'.format(name))
    ax[2].plot(fpr, tpr, color='lightseagreen',label = 'AUC = %0.4f' % roc_auc)
    ax[2].legend(loc = 'lower right')
    ax[2].plot([0, 1], [0, 1],color='coral')
    ax[2].set_xlim([0, 1])
    ax[2].set_ylim([0, 1])
    ax[2].set_ylabel('True Positive Rate')
    ax[2].set_xlabel('False Positive Rate')



    plt.savefig('dl eval {}.png'.format(name))
    plt.show()

def word2vec_tfidf(sentence):
    '''
    This function transforms a sentence to a vector using word2vec vectors and tf-idf as words weights
    '''
    tfidf_vector=vectorizer_en.transform([sentence]).toarray()
    vectors=[]
    for word in sentence.split(' '):
        try:
            score=tfidf_vector[0,vectorizer_en.vocabulary_[word]]
        except:
            score=1
        try:
            vectors.append(word2vec.wv[word]*score)
        except:
            None
    return np.mean(np.array(vectors),axis=0) if len(vectors)>0 else np.array(vectors)


########### THE PROGRAM ############
if __name__ == "__main__":
    df=pd.read_csv('complete_dataset.csv').drop('Unnamed: 0',axis=1)

    lang_chosen='en'


    df_en_id=df.loc[df['langdetect']==lang_chosen,['_id','y','clean3']]
    df_en=df.loc[df['langdetect']==lang_chosen,['y','clean3']]

    df_en1=df_en[df_en['y']==1]
    df_en0=df_en[df_en['y']==0]

    df_en_id1=df_en_id[df_en_id['y']==1]
    df_en_id0=df_en_id[df_en_id['y']==0]

    ## TF-IDF

    vectorizer_en = TfidfVectorizer(min_df=3,max_features=1000)
    vectorizer_en.fit(df_en['clean3'].to_list())
    fited_en_1=vectorizer_en.transform(df_en1['clean3'].to_list())
    fited_en_0=vectorizer_en.transform(df_en0['clean3'].to_list())

    X_train_tfidf,y_train_tfidf,X_test_tfidf,y_test_tfidf=oversample_numpy(fited_en_1.toarray(),
                                                                       fited_en_0.toarray(),df_en1['y'],df_en0['y'])

    # ML
    metrics_tfidf,y_preds_tfidf,y_probs_tfidf=apply_ml_models_text(ml_models,X_train_tfidf,y_train_tfidf,X_test_tfidf,y_test_tfidf)
    rocs_top3(['naivebayes','logmodel','catmodel'],y_test_tfidf,y_probs_tfidf,'tfidf Spanish')

    # DL
    tfidf_trained,tfidf_history_dic,tfidf_preds = apply_dl_models(dl_models,X_train_tfidf,y_train_tfidf,X_test_tfidf,y_test_tfidf,ep=10)
    plot_loss_acc_roc(tfidf_history_dic['rnn'],y_test_tfidf,tfidf_preds['rnn'],'TF-IDF RNN layer Spanish')




    ## WORD2VEC
    num_dim=100
    word2vec = Word2Vec(df_en['clean3'].str.split(' ').to_list(),
                 size= num_dim,
                 seed=101,
                 workers=3,
                 window =3,
                 sg = 1)

    doc2vec_0=list(map(lambda x: word2vec_tfidf(x),df_en0['clean3'].to_list()))
    doc2vec_1=list(map(lambda x: word2vec_tfidf(x),df_en1['clean3'].to_list()))

    y_0=np.array(df_en0['y'].to_list())
    y_1=np.array(df_en1['y'].to_list())

    remove_indices_0=[]

    for i in range(len(doc2vec_0)):
        if doc2vec_0[i].shape[0]!=num_dim:
            remove_indices_0.append(i)
            print(i,doc2vec_0[i].shape)


    remove_indices_1=[]

    for i in range(len(doc2vec_1)):
        if doc2vec_1[i].shape[0]!=num_dim:
            remove_indices_1.append(i)
            print(i,doc2vec_1[i].shape)

    doc2vec_0_rem = [i for j, i in enumerate(doc2vec_0) if j not in remove_indices_0]
    y_0_rem = [i for j, i in enumerate(y_0) if j not in remove_indices_0]

    doc2vec_1_rem = [i for j, i in enumerate(doc2vec_1) if j not in remove_indices_1]
    y_1_rem = [i for j, i in enumerate(y_1) if j not in remove_indices_1]

    doc2vec0=np.concatenate(doc2vec_0_rem).reshape(len(doc2vec_0_rem),num_dim)
    doc2vec1=np.concatenate(doc2vec_1_rem).reshape(len(doc2vec_1_rem),num_dim)

    X_train_w2v,y_train_w2v,X_test_w2v,y_test_w2v=oversample_numpy(doc2vec1,doc2vec0,pd.Series(y_1_rem),pd.Series(y_0))

    # ML
    metrics_w2v,y_preds_w2v,y_probs_w2v=apply_ml_models_text(ml_models,X_train_w2v,y_train_w2v,X_test_w2v,y_test_w2v)
    rocs_top3(['catmodel','lightmodel','xgbmodel'],y_test_w2v,y_probs_w2v,'w2v Spanish')

    # DL
    w2v_trained,w2v_history_dic,w2v_preds = apply_dl_models(dl_models,X_train_w2v,y_train_w2v,X_test_w2v,y_test_w2v,ep=10) 
    plot_loss_acc_roc(w2v_history_dic['perceptron'],y_test_w2v,w2v_preds['perceptron'],'W2V Basic NN Spanish')



    ## BERT PRE-TRAINED (the file is not included because it's 1,6gb)
    with open('bert_embeddings.json', 'r') as f:
        bert_dict = json.load(f)

    bert_1=np.array([bert_dict[item] for item in df_en_id1['_id'].to_list()])
    bert_0=np.array([bert_dict[item] for item in df_en_id0['_id'].to_list()])

    X_train_bert,y_train_bert,X_test_bert,y_test_bert=oversample_numpy(bert_1,bert_0,df_en_id1['y'],df_en_id0['y'])

    # ML
    metrics_bert,y_preds_bert,y_probs_bert=apply_ml_models_text(ml_models,X_train_bert,y_train_bert,X_test_bert,y_test_bert)
    rocs_top3(['logmodel','catmodel','lightmodel'],y_test_bert,y_probs_bert,'bert Spanish')

    # DL
    bert_trained,bert_history_dic,bert_preds = apply_dl_models(dl_models,X_train_bert,y_train_bert,X_test_bert,y_test_bert,ep=10) 
    plot_loss_acc_roc(bert_history_dic['perceptron'],y_test_bert,bert_preds['perceptron'],'BERT Basic NN Spanish')






