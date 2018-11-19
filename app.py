from flask import Flask, abort, render_template, jsonify, request
#from api import find_book
import numpy as np
import pandas as pd
import pickle
import dill
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from nltk import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# setup parameters
RSEED = 0
bow = 'tf' # tf,tfid
stem_type = 'lemma' # snow, lemma
n_gram = '1gm' # 1gm or 2gm
topic_model = 'lda' #lda, nmf
piece = '/Users/user/Documents/Data_Science/Github/BookReccomender2/input_sample.txt'

# opening the pipeline
vectorizer = dill.load(open('/Users/user/Documents/Data_Science/Github/BookReccomender2/data/vectors/'+'tf_vectorizer_lemma_1gm', 'rb'))
wordnet_lemmatizer = WordNetLemmatizer()
snow = SnowballStemmer('english')
model = dill.load(open('/Users/user/Documents/Data_Science/Github/BookReccomender2/data/vectors/'+ topic_model +'_'+ stem_type + '_' + n_gram,'rb'))
min_max_scaler = dill.load(open('/Users/user/Documents/Data_Science/Github/BookReccomender2/data/vectors/scaler','rb'))

def find_book(features):


    #piece ='./data/samples/fifty_shades.txt'
    piece_str = features['excerpt']
    print(piece_str)
    fout = open(piece,'w')
    fout.write(piece_str)
    fout.close()


    # vectorize + topic model
    vector = vectorizer.transform([piece])
    topic_vector = model.transform(vector)
    df_excerpt_a = pd.DataFrame(topic_vector, columns=['topic_'+ str(i)for i in range(1,21)])

    # sentiment stuff
    book_excerpt = TextBlob(piece)
    word_count = len(book_excerpt.words)
    sentence_count =len(book_excerpt.sentences)
    avg_len = word_count/sentence_count
    sentiment_excerpt = [[word_count,sentence_count,avg_len,book_excerpt.sentiment[0],book_excerpt.sentiment[1]]]
    df_excerpt_b = pd.DataFrame(sentiment_excerpt, \
                            columns = ['word_count','sentence_count','sentence_length','polarity','subjectivity'])
    #join
    df_excerpt = pd.concat([df_excerpt_a,df_excerpt_b], axis=1)

    # tranformations

    # log transform the counts
    column_names_to_log_1 = ['word_count', 'sentence_count', 'sentence_length']
    df_excerpt.loc[:,column_names_to_log_1] = df_excerpt.loc[:,column_names_to_log_1].apply(np.log)


    # normalize subjectivity and polarity
    column_names_to_normalize = ['subjectivity', 'polarity']
    x = df_excerpt[column_names_to_normalize].values
    x_scaled = min_max_scaler.transform(x) # only transform
    df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df_excerpt.index)
    df_excerpt[column_names_to_normalize] = df_temp

    # log transform topics
    df_excerpt.loc[:,'topic_1':'topic_20'] = df_excerpt.loc[:,'topic_1':'topic_20'].apply(np.log)

    # load the mother load aka the corpus vector
    #import os

    #print("Path at terminal when executing this file")
    #print(os.getcwd() + "\n")
    corpus = pd.read_csv('/Users/user/Documents/Data_Science/Github/BookReccomender2/data/final_full.csv')
    corpus = corpus.drop(columns ='Unnamed: 0')

    # columns to drop before finding similarity
    drop_cols =['word_count','sentence_count']
    corpus = corpus.drop(columns =drop_cols)
    df_excerpt = df_excerpt.drop(columns = drop_cols)

    # shape em up to numpy arrays
    given_excerpt = np.array(df_excerpt)
    search_in = np.array(corpus.iloc[:,3:])

    # cosine cosine_similarity
    results = cosine_similarity(search_in, given_excerpt)
    show_me = pd.DataFrame(results).sort_values(0, ascending=False).head(1)
    title = corpus.iloc[list(show_me.index),0:2].values
    similar_book = title[0][0] + ' by '+ title[0][1]



    result = {
        'similar_book': similar_book,
    }
    return result

app = Flask('predict')

@app.route('/predict', methods=['POST'])
def go_find():
    if not request.json:
        #console.log('MMMMM')
        abort(400)
    data = request.json

    response =find_book(data)

    return jsonify(response)

@app.route('/')
def index():
    return render_template('index.html')

app.run()
