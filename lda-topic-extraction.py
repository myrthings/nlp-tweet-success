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

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaMulticore,CoherenceModel, Word2Vec
from wordcloud import WordCloud
from sklearn.manifold import TSNE


plt.style.use('seaborn')

## from a medium guy
def lda_move_topics(dictionary, corpus, texts, limit, start=2,step=1):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    perplexity_values : Perplexity values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    perplexity_values = []
    model_list = []
    for num_topics in range(start, limit,step):
        model = LdaMulticore(corpus,
                       num_topics=num_topics,
                       id2word=dictionary,
                       passes=2,
                       workers=2)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model,
                                        texts=texts,
                                        dictionary=dictionary,
                                        coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        perplexity_values.append(model.log_perplexity(corpus))

    return model_list, coherence_values, perplexity_values

# given lda this function transforms it into a dataframe
def format_topics_sentences(ldamodel, corpus,ids):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), ids[i]]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Tuit_ID']

    # Add original text to the end of the output
    return(sent_topics_df)

## from another guy
def display_closestwords_tsnescatterplot(model, word, size,num_words):
    
    arr = np.empty((0,size), dtype='f')
    word_labels = [word]
    close_words = model.similar_by_word(word,topn=num_words)
    arr = np.append(arr, np.array([model[word]]), axis=0)
    
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    
    plt.scatter(x_coords, y_coords)
    
    for label, x, y in zip(word_labels, x_coords, y_coords):
        if label==word:
            plt.scatter([x],[y],color='red')
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',fontsize=12)
    
    plt.xlim(x_coords.min()+0.0005, x_coords.max()+0.0005)
    plt.ylim(y_coords.min()+0.0005, y_coords.max()+0.0005)
    plt.show()



if __name__ == "__main__":
    df=pd.read_csv('texts_clean3.csv').drop(['Unnamed: 0'],axis=1) #the data uploaded works too

    lang_chosen='ca'
    df_en=df[df['langdetect']==lang_chosen]

    # get only the texts with more than 20 chars
    df_en=df_en[df_en['clean3'].fillna(' ').apply(lambda x: len(x)>20)]

    # create the dictionary, corpus and tfidf with gensim
    texts_en=df_en['clean3'].str.split(' ').to_list()
    dictionary_en = Dictionary(texts_en)

    dictionary_en.filter_extremes(no_below=15, no_above=0.5, keep_n=10000)
    bow_corpus_en = [dictionary_en.doc2bow(doc) for doc in texts_en]

    tfidf = TfidfModel(bow_corpus_en)
    corpus_tfidf_en = tfidf[bow_corpus_en]


    # get the evaluation for various topics and chose the best number of them

    limit=100; start=2; step=10;
    model_list, coherence_values, perplexity_values = lda_move_topics(dictionary=dictionary_en,
                                                        corpus=corpus_tfidf_en,
                                                        texts=texts_en,
                                                        start=start,
                                                        limit=limit,
                                                       step=step)

    # plot coherence an perplexity
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
    x = range(start, limit,step)
    ax1.plot(x, coherence_values,label='coherence_values')
    ax2.plot(x, perplexity_values,label='perplexity_values')
    ax1.set_xlabel("Num Topics")
    ax1.set_ylabel("Coherence score")
    ax1.set_ylim(ymin=0)
    ax2.set_xlabel("Num Topics")
    ax2.set_ylabel("Perplexity score")
    ax2.set_ylim(ymax=0)
    ax1.legend()
    ax2.legend()
    fig.suptitle('TFIDF CORPUS CA')
    plt.show()

    # really slow, there're better ways to do that
    corpus_matrix=[]
    for idx,line in enumerate(corpus_tfidf_en.corpus):
        matrix=pd.DataFrame(line,columns=['index',idx]).set_index('index').T
        corpus_matrix.append(matrix)
    corpus_matrix_en=pd.concat(corpus_matrix,axis=0)
    corpus_matrix_en=corpus_matrix_en.fillna(0)
    corpus_matrix_en.columns=list(dictionary_en.token2id.keys())

    # reduce the dimensionaly to plot it
    tsne = TSNE(n_components=2, verbose=1, n_iter=300,random_state=0)
    tsne_results = tsne.fit_transform(corpus_matrix_en)


    # plot the tsne
    corpus_matrix_en['tsne-2d-one'] = tsne_results[:,0]
    corpus_matrix_en['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        palette=sns.color_palette("hls", 10),
        data=corpus_matrix_en,
        legend="full",
        alpha=0.3
    )



    # final model chosen
    lda_tfidf_en = LdaMulticore(corpus_tfidf_en,
                           num_topics=72,
                           id2word=dictionary_en,
                           passes=2,
                           workers=2)

    df_topic_en = format_topics_sentences(ldamodel=lda_tfidf_en,
                                                      corpus=corpus_tfidf_en,
                                                        ids=df_en['_id'].to_list())

    # plots the tsne with different colors for every topic
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue='Dominant_Topic',
        palette=sns.color_palette("hls", 10),
        data=topics_en,
        legend="full",
        alpha=0.8
    )
    #plt.xlim(xmin=-10,xmax=10)
    #plt.ylim(ymin=-10,ymax=10)
    plt.show()


    ## see how every topic looks like
    for item in topics_en['Dominant_Topic'].sort_values().unique():
        part=topics_en[topics_en['Dominant_Topic']==item]
        
        print('\n Topic:',item,'with',len(part),'elements',round(len(part)/len(topics_en)*100,2),'%')
        print(lda_tfidf_en.print_topic(item, 10))
        
        ##wordcloud
        wordcloud = WordCloud(max_font_size=50,
                          max_words=100,
                          background_color="white").generate(' '.join(part['clean3'].to_list()))
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        
        ##word2vec tsne
        model = Word2Vec((part['clean3'].str.split(' ')).to_list(),
                     size= 100,
                     seed=0,
                     workers=3,
                     window =3,
                     sg = 1)
        display_closestwords_tsnescatterplot(model, 'catalunya', 100,20) 



