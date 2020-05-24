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
import emoji
import re
from nltk.corpus import stopwords
import requests
import stanfordnlp

def drop_emoji(line):
    for emoj in emoji.UNICODE_EMOJI.keys():
        line=line.replace(emoj,' ')
    return line

def remove_all_punctuation(s):
    for c in string.punctuation+''.join(['–','—','►','…','”','“','¿','¡','»','«','·','•','‘','’']):
        s=s.replace(c," ")
    return s


#english specific
ap_en_dic={"won\'t":"will not",
       "can\'t":"can not",
       "n\'t":" not",
       "\'re":" are",
       "\'s":" is",
       "\'d":" would",
       "\'ll":" will",
       "\'t":" not",
       "\'ve":" have",
       "\'m":" am",
       "won\’t":"will not",
       "can\’t":"can not",
       "n\’t":" not",
       "\’re":" are",
       "\’s":" is",
       "\’d":" would",
       "\’ll":" will",
       "\’t":" not",
       "\’ve":" have",
       "\’m":" am"}

ap_ca_dic={" l'":' el ',
          " d'":' de ',
          " s'":' se ',
          " m'":' me ',
          " n'":' nos ',
          "'n ":' nos ',
          " l’":' el ',
          " d’":' de ',
          " s’":' se ',
          " m’":' me ',
          " n’":' nos ',
          "’n ":' nos '}

def remove_ap(line,dic):
    for contraction in dic.keys():
        line=re.sub(contraction,dic[contraction],line)
    return line


#spanish specific
cont_dic={'q':'que',
         'tb':'también'}

def change_words(words):
    return list(map(lambda x: cont_dic[x] if x in cont_dic.keys() else x,words))

def lemmat(words):
    lems=[]
    doc = nlp(words)
    for sent in doc.sentences:
        for word in sent.words:
            lems.append(word.lemma)
    return lems

def lemma_list(words):
    lems=[]
    doc = nlp(words)
    for sent in doc.sentences:
        try:
            lem=[]
            for word in sent.words:
                lem.append(word.lemma)
            lems.append(lem)
        except:
            lems.append([])
    return lems

def remove_stop(words,stopw):
    return list(set(filter(lambda x: x not in stopw+[''],words)))

if __name__ == "__main__":

    txt=pd.read_csv('text_tuits.csv').drop(['Unnamed: 0'],axis=1)

    ## clean1
    txt['text']=txt['text'].fillna(' ')
    txt['clean1']=txt['text'].apply(lambda x: x.lower().encode("utf8",'ignore').decode("utf8"))
    txt['clean1']=txt['clean1'].apply(lambda x: drop_emoji(x))
    txt['clean1']=txt['clean1'].str.replace('http\S+|www.\S+|pic.twitter.\S+|@\S+|#|\n|\r', ' ', case=False)
    txt_en=txt[txt['langdetect']=='en']
    txt_es=txt[txt['langdetect']=='es']
    txt_ca=txt[txt['langdetect']=='ca']


    txt_en['clean1']=txt_en['clean1'].apply(lambda x: remove_ap(x,ap_en_dic))
    txt_ca['clean1']=txt_ca['clean1'].apply(lambda x: remove_ap(x,ap_ca_dic))

    txt_en['clean1']=txt_en['clean1'].apply(lambda x: remove_all_punctuation(x))
    txt_es['clean1']=txt_es['clean1'].apply(lambda x: remove_all_punctuation(x))
    txt_ca['clean1']=txt_ca['clean1'].apply(lambda x: remove_all_punctuation(x))

    txt_en['clean1']=txt_en['clean1'].apply(lambda x: ' '.join(list(filter(lambda x: x!='',x.split(' ')))))
    txt_es['clean1']=txt_es['clean1'].apply(lambda x: ' '.join(change_words(list(filter(lambda x: x!='',x.split(' '))))))
    txt_ca['clean1']=txt_ca['clean1'].apply(lambda x: ' '.join(change_words(list(filter(lambda x: x!='',x.split(' '))))))


    txt_en=txt_en[txt_en['clean1'].apply(lambda x: len(x)>20)]
    txt_es=txt_es[txt_es['clean1'].apply(lambda x: len(x)>20)]
    txt_ca=txt_ca[txt_ca['clean1'].apply(lambda x: len(x)>20)]

    #stanfordnlp.download("en")
    #stanfordnlp.download("es")
    #stanfordnlp.download("ca")


    ## clean2
    # en
    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma',
                           lang='en',
                          tokenize_pretokenized=True)

    tokens_en=(txt_en['clean1'].str.split(' ')).to_list()
    list_en=lemma_list(tokens_en)

    txt_en['clean2']=list_en

    # es
    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma',
                           lang='es',
                          tokenize_pretokenized=True)

    tokens_es=(txt_es['clean1'].str.split(' ')).to_list()
    list_es=lemma_list(tokens_es)

    txt_es['clean2']=list_es

    # ca
    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma',
                           lang='ca',
                          tokenize_pretokenized=True)

    tokens_ca=(txt_ca['clean1'].str.split(' ')).to_list()
    list_ca=lemma_list(tokens_ca)

    txt_ca['clean2']=list_ca

    # all
    txt_new=pd.concat([txt_en,txt_es,txt_ca],axis=0)
    txt_new['clean2']=txt_new['clean2'].apply(lambda x: ' '.join(x))

    ntxt_en=txt_new[txt_new['langdetect']=='en']
    ntxt_es=txt_new[txt_new['langdetect']=='es']
    ntxt_ca=txt_new[txt_new['langdetect']=='ca']


    ## clean3
    stop_en=stopwords.words('english')
    stop_es=stopwords.words('spanish')
    cat_stopwords='https://raw.githubusercontent.com/stopwords-iso/stopwords-ca/master/stopwords-ca.txt'
    stop_ca=requests.get(cat_stopwords).text.split('\n')
    other_stop=['rt','retweet','quote','fav','via','vía','wtf','omg','xd','idk','lol','imho','lmao','btw','']

    ntxt_en['words']=ntxt_en['clean2'].apply(lambda x: remove_stop(x.split(' '),stop_en+other_stop))
    ntxt_es['words']=ntxt_es['clean2'].apply(lambda x: remove_stop(x.split(' '),stop_es+other_stop))
    ntxt_ca['words']=ntxt_ca['clean2'].apply(lambda x: remove_stop(x.split(' '),stop_ca+other_stop))
    ntxt_en['clean3']=ntxt_en['words'].apply(lambda x: ' '.join(x))
    ntxt_es['clean3']=ntxt_es['words'].apply(lambda x: ' '.join(x))
    ntxt_ca['clean3']=ntxt_ca['words'].apply(lambda x: ' '.join(x))


    txt2=pd.concat([ntxt_en,ntxt_es,ntxt_ca],axis=0)








