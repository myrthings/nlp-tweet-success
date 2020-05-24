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
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

from matplotlib import rc

# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':12})

# Set the font used for MathJax - more on this later
rc('mathtext',**{'default':'regular'})


def group_data(data,col,target):
    df=deepcopy(data)
    group=df.groupby([col,target])[['_id']].count().reset_index().set_index(col)
    group['total']=group.groupby([col])[['_id']].sum()
    group=group.reset_index().set_index(target)
    group['totalcat']=group.groupby([target])[['_id']].sum()
    group['perc']=group['_id']/group['total']
    group['perccat']=group['_id']/group['totalcat']
    group=group.reset_index().set_index([col,target])
    return group

def plot_bars2(df,col,ax,cat_dic=None):
    axc=ax.twinx()
    bars=df['perc'].unstack()
    bars.columns=['no','yes']
    bars.plot(kind='bar',
              width=0.8,
              color=['lightblue','lightcoral'],
              ax=ax)

    
    line=df.reset_index().set_index(col)[['total']].drop_duplicates()
    line['perc']=line['total']/line['total'].sum()
    if -1 in line.index:
        x=ax.get_xticks()
        axc.plot(x,line['perc'], #para arreglar que uno los toma como números y otros no 
                      ls='--',
                      marker='o',
                      color='grey',
                      label='total')
    else:
        line['perc'].plot(kind='line', #para arreglar que uno los toma como números y otros no 
                      ls='--',
                      marker='o',
                      color='grey',
                      label='total')

    xticks=ax.get_xticks()
    ax.set_xlim(xmin=min(xticks)-0.5,xmax=max(xticks)+0.5)
    
    ax.set_ylim(ymin=0,ymax=1)
    axc.set_ylim(ymin=0,ymax=1)
    
    ax.legend(title='Tweet Success',loc='upper left')
    axc.legend(loc='upper right')
    
    ax.set_ylabel('Distribution inside the value (%)',fontsize=14)
    axc.set_ylabel('Distribution of values (%)',fontsize=14)
    
    yticks=ax.get_yticks()
    ax.set_yticklabels(['{:0.0%}'.format(lab) for lab in yticks])
    
    yticks=axc.get_yticks()
    axc.set_yticklabels(['{:0.0%}'.format(lab) for lab in yticks])
    
    if cat_dic:
        ax.set_xticklabels(cat_dic)
    
    ax.spines['top'].set_visible(False)
    axc.spines['top'].set_visible(False)
    

def plot_piramid(df,col,ax,cat_dic=None):
    bars=df['perccat'].unstack()
    bars.columns=['no','yes']
    bars['yes']=-bars['yes']
    bars['yes'].plot(kind='barh',
              width=0.9,
              color='lightblue',
              ax=ax)
    
    bars['no'].plot(kind='barh',
              width=0.9,
              color='lightcoral',
              ax=ax)
    
    
    ax.set_xlim(xmin=-1,xmax=1)
    
    yticks=ax.get_yticks()
    ax.vlines(0,yticks.min()-0.5,yticks.max()+0.5,linewidth=0.5)
    ax.set_ylim(ymin=yticks.min()-0.5,ymax=yticks.max()+0.5)
    ax.text(-0.8,yticks.max(),'No-Success\nDistribution',va='top')
    ax.text(0.8,yticks.max(),'Success\nDistribution',ha='right',va='top')

    ax.set_xlabel('Distribution of success (%)',fontsize=14)
    
    xticks=ax.get_xticks()
    ax.set_xticklabels(['{:0.0%}'.format(lab if lab>=0 else -lab) for lab in xticks])
    
    if cat_dic:
        ax.set_yticklabels(cat_dic)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    







if __name__ == "__main__":
    all_cat=pd.read_csv('complete_dataset.csv').drop('Unnamed: 0',axis=1)


    ## PLOT TUITS DISTRIBUTIONS
    fig,ax=plt.subplots(1,2,figsize=(18,5))
    rsdf=pd.DataFrame(all_cat['RS_cat'].astype(int).value_counts())
    rsdf['perc']=rsdf['RS_cat']/rsdf['RS_cat'].sum()
    rsdf['perc'].plot(kind='bar',color='lightblue',ax=ax[0])

    for i,value in enumerate(rsdf['perc'].to_list()):
        ax[0].text(i,value+0.025,'{:1.2%}'.format(value),ha='center')

    ax[0].set_ylim(ymin=0,ymax=1)
    yticks=ax[0].get_yticks()
    ax[0].set_yticklabels(['{:1.0%}'.format(l) for l in yticks])

    ax[0].set_xlabel('RS Category')
    ax[0].set_ylabel('Distribution of tweets (%)')

    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].set_title('RS categories distribution',fontsize=14)


    #----------

    rsdf=pd.DataFrame(all_cat['y'].dropna().astype(int).value_counts())
    rsdf['perc']=rsdf['y']/rsdf['y'].sum()
    rsdf['perc'].plot(kind='bar',color='lightcoral',ax=ax[1])

    for i,value in enumerate(rsdf['perc'].to_list()):
        ax[1].text(i,value+0.025,'{:1.2%}'.format(value),ha='center')

    ax[1].set_ylim(ymin=0,ymax=1)
    yticks=ax[1].get_yticks()
    ax[1].set_yticklabels(['{:1.0%}'.format(l) for l in yticks])

    ax[1].set_xlabel('Target (y)')
    ax[1].set_ylabel('Distribution of tweets (%)')

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].set_title('Target distribution',fontsize=14)

    plt.savefig('RS and Target distribution.png')
    plt.show()

    ## PLOT ALL BARS
    lista=['langdetect','tuit_topic','num_websites','num_mentions','num_hashtags', 'num_emojis',
           'is_reply', 'is_quote','created_day', 'created_hour', ('num_franja','franja'),'media_photo','media_video', 'media_gif',
          'user_createdyear', 'user_descrip_lang',
          ('num_user_age','user_age'),('num_influence','influence'),
           ('num_exclusivity','exclusivity'),('num_tuitero','tuitero'),'user_topic']

    for i,item in enumerate(lista):
        if len(item)==2:
            labels=all_cat[[item[0],item[1]]].drop_duplicates().dropna().sort_values(item[0])[item[1]].to_list()
            item=item[0]
        else:
            labels=None
            
        if i%2==0:
            fig,ax=plt.subplots(1,2,figsize=(15,5))
        dat=group_data(all_cat,item,'y')
        plot_bars2(dat,item,ax[i%2],labels)
        
        if i%2==1:
            plt.tight_layout(pad=0.4)
            plt.savefig('chart {} {} {}.png'.format(i,lista[i-1],lista[i]))
            plt.show()


    ## PLOT ALL PIRAMIDS
    lista=['langdetect','tuit_topic','num_websites','num_mentions','num_hashtags', 'num_emojis',
           'is_reply', 'is_quote','created_day', 'created_hour', ('num_franja','franja'),'media_photo','media_video', 'media_gif',
          'user_createdyear', 'user_descrip_lang',
          ('num_user_age','user_age'),('num_influence','influence'),
           ('num_exclusivity','exclusivity'),('num_tuitero','tuitero'),'user_topic']

    for i,item in enumerate(lista):
        if len(item)==2:
            labels=all_cat[[item[0],item[1]]].drop_duplicates().dropna().sort_values(item[0])[item[1]].to_list()
            item=item[0]
        else:
            labels=None

        if i%2==0:
            fig,ax=plt.subplots(1,2,figsize=(15,5))
        dat=group_data(all_cat,item,'y')
        plot_piramid(dat,item,ax[i%2],labels)
        
        if i%2==1:
            plt.tight_layout(pad=0.4)
            plt.savefig('piramid {} {} {} .png'.format(i,lista[i-1],lista[i]))
            plt.show()


    ## PLOT ALL PIRAMID-BARS PAIRS
    lista=['langdetect','tuit_topic','num_websites','num_mentions','num_hashtags', 'num_emojis',
           'is_reply', 'is_quote','created_day', 'created_hour', ('num_franja','franja'),'media_photo','media_video', 'media_gif',
          'user_createdyear', 'user_descrip_lang',
          ('num_user_age','user_age'),('num_influence','influence'),
           ('num_exclusivity','exclusivity'),('num_tuitero','tuitero'),'user_topic','num_words','num_chars']

    for i,item in enumerate(lista):
        if len(item)==2:
            labels=all_cat[[item[0],item[1]]].drop_duplicates().dropna().sort_values(item[0])[item[1]].to_list()
            item=item[0]
        else:
            labels=None
        
        fig,ax=plt.subplots(1,2,gridspec_kw={'width_ratios':[3, 4]},figsize=(15,5))
        dat=group_data(all_cat,item,'y')
        plot_piramid(dat,item,ax[0],labels)
        plot_bars2(dat,item,ax[1],labels)
        plt.tight_layout(pad=0.4)
        plt.savefig('figure {} {}.png'.format(i,item))
        plt.show()


    ## PLOT HISTOGRAMS
    fig,ax=plt.subplots(1,2,figsize=(15,4))
    all_cat.loc[all_cat['y']==1,'num_words'].plot(kind='hist',
                                                  density=True,
                                                  bins=15,
                                                 color='lightcoral',
                                                 alpha=0.5,
                                                  label='Success',
                                                 ax=ax[0])
    all_cat.loc[all_cat['y']==0,'num_words'].plot(kind='hist',
                                                  density=True,
                                                  bins=15,
                                                 color='lightblue',
                                                 alpha=0.5,
                                                  label='No-Success',
                                                 ax=ax[0])


    all_cat.loc[all_cat['y']==1,'num_chars'].plot(kind='hist',
                                                  density=True,
                                                  bins=20,
                                                 color='lightcoral',
                                                 alpha=0.5,
                                                  label='Success',
                                                 ax=ax[1])
    all_cat.loc[all_cat['y']==0,'num_chars'].plot(kind='hist',
                                                  density=True,
                                                  bins=20,
                                                 color='lightblue',
                                                 alpha=0.5,
                                                  label='No-Success',
                                                 ax=ax[1])

    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')

    yticks=ax[0].get_yticks()
    ax[0].set_yticklabels(['{:0.0%}'.format(lab) for lab in yticks])

    yticks=ax[1].get_yticks()
    ax[1].set_yticklabels(['{:0.1%}'.format(lab) for lab in yticks])

    ax[0].set_xlabel('num_words')
    ax[1].set_xlabel('num_chars')

    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    plt.savefig('Words Chars distrib.png')
    plt.show()


    ## PLOT WORDS SCATTERS
    ca_tsne=all_cat.loc[all_cat['langdetect']=='ca',['y','tuit_topic','tuit_tsne_x','tuit_tsne_y']].rename(columns={'tuit_tsne_x':'tfidf_tsne_x','tuit_tsne_y':'tfidf_tsne_y'})
    en_tsne=all_cat.loc[all_cat['langdetect']=='en',['y','tuit_topic','tuit_tsne_x','tuit_tsne_y']].rename(columns={'tuit_tsne_x':'tfidf_tsne_x','tuit_tsne_y':'tfidf_tsne_y'})
    es_tsne=all_cat.loc[all_cat['langdetect']=='es',['y','tuit_topic','tuit_tsne_x','tuit_tsne_y']].rename(columns={'tuit_tsne_x':'tfidf_tsne_x','tuit_tsne_y':'tfidf_tsne_y'})

    fig,ax=plt.subplots(1,3,figsize=(20,5))
    ca_tsne.loc[ca_tsne['y']==0].plot(kind='scatter',
                                                                    x='tfidf_tsne_x',
                                                                    y='tfidf_tsne_y',
                                                                    color='lightblue',
                                                                    label='No-Success',
                                                                   ax=ax[0])

    ca_tsne.loc[ca_tsne['y']==1].plot(kind='scatter',
                                                                    x='tfidf_tsne_x',
                                                                    y='tfidf_tsne_y',
                                                                    color='lightcoral',
                                                                    label='Success',
                                                                   ax=ax[0])
    ax[0].legend(loc='upper left')
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].set_title('Catalan Tweets ({:1.2%} of the data)'.format(len(ca_tsne['y'].dropna())/len(all_cat['y'].dropna())))

    en_tsne.loc[en_tsne['y']==0].plot(kind='scatter',
                                                                    x='tfidf_tsne_x',
                                                                    y='tfidf_tsne_y',
                                                                    color='lightblue',
                                                                    label='No-Success',
                                                                   ax=ax[1])

    en_tsne.loc[en_tsne['y']==1].plot(kind='scatter',
                                                                    x='tfidf_tsne_x',
                                                                    y='tfidf_tsne_y',
                                                                    color='lightcoral',
                                                                    label='Success',
                                                                   ax=ax[1])
    ax[1].legend(loc='upper left')
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].set_title('English Tweets ({:1.2%} of the data)'.format(len(en_tsne['y'].dropna())/len(all_cat['y'].dropna())))


    es_tsne.loc[es_tsne['y']==0].plot(kind='scatter',
                                                                    x='tfidf_tsne_x',
                                                                    y='tfidf_tsne_y',
                                                                    color='lightblue',
                                                                    label='No-Success',
                                                                   ax=ax[2])

    es_tsne.loc[es_tsne['y']==1].plot(kind='scatter',
                                                                    x='tfidf_tsne_x',
                                                                    y='tfidf_tsne_y',
                                                                    color='lightcoral',
                                                                    label='Success',
                                                                   ax=ax[2])
    ax[2].legend(loc='upper left')
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].set_title('Spanish Tweets ({:1.2%} of the data)'.format(len(es_tsne['y'].dropna())/len(all_cat['y'].dropna())))

    plt.savefig('TFIDF-TSNE clouds.png')
    plt.show()



    ## PLOT WORDCLOUDS    
    fig,ax=plt.subplots(1,3,figsize=(15,10))
    wordcloud = WordCloud(max_font_size=50,
                          max_words=100,
                          width=400,
                          height=250,
                          background_color="white",
                         ).generate(' '.join(textdf.loc[textdf['langdetect']=='ca','clean3'].to_list()))

    ax[0].imshow(wordcloud,interpolation="bilinear")

    wordcloud = WordCloud(max_font_size=50,
                          max_words=100,
                          width=400,
                          height=250,
                          background_color="white",
                         ).generate(' '.join(textdf.loc[textdf['langdetect']=='en','clean3'].to_list()))

    ax[1].imshow(wordcloud,interpolation="bilinear")

    wordcloud = WordCloud(max_font_size=50,
                          max_words=100,
                          width=400,
                          height=250,
                          background_color="white",
                         ).generate(' '.join(textdf.loc[textdf['langdetect']=='es','clean3'].to_list()))

    ax[2].imshow(wordcloud,interpolation="bilinear")

    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")

    ax[0].set_title('Catalan Wordcloud',fontsize=14)
    ax[1].set_title('English Wordcloud',fontsize=14)
    ax[2].set_title('Spanish Wordcloud',fontsize=14)

    plt.tight_layout()
    plt.savefig('Wordclouds.png')
    plt.show()




    ## PLOTS WORDS HIGH FREQ
    fig,ax=plt.subplots(2,3,figsize=(18,10),constrained_layout=True)

    #-------------------
    ca_words_n.value_counts().sort_values().tail(15).plot(kind='barh',
                                                    color='lightblue',
                                                    ax=ax[0,0])
    ax[0,0].spines['top'].set_visible(False)
    ax[0,0].spines['right'].set_visible(False)
    ax[0,0].set_xlabel('Word frequency')
    ax[0,0].set_title('Catalan no-success tweets.\nTop 15 Words with the highest frequency')

    ca_words_y.value_counts().sort_values().tail(15).plot(kind='barh',
                                                    color='lightcoral',
                                                    ax=ax[1,0])
    ax[1,0].spines['top'].set_visible(False)
    ax[1,0].spines['right'].set_visible(False)
    ax[1,0].set_xlabel('Word frequency')
    ax[1,0].set_title('Catalan success tweets.\nTop 15 Words with the highest frequency')

    #-------------------

    en_words_n.value_counts().sort_values().tail(15).plot(kind='barh',
                                                    color='lightblue',
                                                    ax=ax[0,1])
    ax[0,1].spines['top'].set_visible(False)
    ax[0,1].spines['right'].set_visible(False)
    ax[0,1].set_xlabel('Word frequency')
    ax[0,1].set_title('English no-success tweets.\nTop 15 Words with the highest frequency')

    en_words_y.value_counts().sort_values().tail(15).plot(kind='barh',
                                                    color='lightcoral',
                                                    ax=ax[1,1])
    ax[1,1].spines['top'].set_visible(False)
    ax[1,1].spines['right'].set_visible(False)
    ax[1,1].set_xlabel('Word frequency')
    ax[1,1].set_title('English success tweets.\nTop 15 Words with the highest frequency')

    #-------------------

    es_words_n.value_counts().sort_values().tail(15).plot(kind='barh',
                                                    color='lightblue',
                                                    ax=ax[0,2])
    ax[0,2].spines['top'].set_visible(False)
    ax[0,2].spines['right'].set_visible(False)
    ax[0,2].set_xlabel('Word frequency')
    ax[0,2].set_title('Spanish no-success tweets.\nTop 15 Words with the highest frequency')

    es_words_y.value_counts().sort_values().tail(15).plot(kind='barh',
                                                    color='lightcoral',
                                                    ax=ax[1,2])
    ax[1,2].spines['top'].set_visible(False)
    ax[1,2].spines['right'].set_visible(False)
    ax[1,2].set_xlabel('Word frequency')
    ax[1,2].set_title('Spanish success tweets.\nTop 15 Words with the highest frequency')

    plt.savefig('Words success high freq.png')
    plt.show()












