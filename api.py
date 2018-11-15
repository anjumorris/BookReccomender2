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

# opening the pipeline
vectorizer = dill.load(open('./data/vectors/'+'tf_vectorizer_lemma_1gm', 'rb'))
wordnet_lemmatizer = WordNetLemmatizer()
snow = SnowballStemmer('english')
model = dill.load(open('./data/vectors/'+ topic_model +'_'+ stem_type + '_' + n_gram,'rb'))
min_max_scaler = dill.load(open('./data/vectors/scaler','rb'))




#c_b = {}
example = {
  'excerpt': '“Does this mean you’re going to make love to me tonight, Christian?” Holy shit. Did I just say that? His mouth drops open slightly, but he recovers quickly. “No, Anastasia it doesn’t. Firstly, I don’t make love. I fuck… hard. Secondly, there’s a lot more paperwork to do, and thirdly, you don’t yet know what you’re in for. You could still run for the hills. Come, I want to show you my playroom. ”My mouth drops open. Fuck hard! Holy shit, that sounds so… hot. But why are we looking at a playroom? I am mystified. “You want to play on your Xbox?” I ask. He laughs, loudly.“No, Anastasia, no Xbox, no Playstation. Come.”… Producing a key from his pocket, he unlocks yetanother door and takes a deep breath. “You can leave anytime. The helicopter is on stand-by to take you whenever you want to go, you can stay the night and go home in the morning. It’s fine whatever you decide.”“Just open the damn door, Christian. ”He opens the door and stands back to let me in. I gaze at him once more. I so want to know what’s inhere. Taking a deep breath I walk in. And it feels I’ve time-traveled back to the sixteenth century and the Spanish Inquisition. Holy fuck.“I nod, wide-eyed, my heart bouncing off my ribs, my blood thundering around my body.Hereaches down, and from his pants pocket, he takes out his silver grey silk tie… that silver grey woven tie that leaves small impressions of its weave on my skin. He moves so quickly, sitting astride me as he fastens my wrists together, but this time, he ties the other end of the tie to one of the spokes of my white iron headboard. He pulls at my binding, checking it’s secure. I’m not going anywhere. I’m tied, literally, to my bed, and I’m so aroused.He slides off me and stands beside the bed, staring down at me, his eyes dark with want. His look is triumphant, mixed with relief.“We’re going to rectify the situation right now.” “What do you mean? What situation?” “Your situation. Ana, I’m going to make love to you, now.” “Oh.” The floor has fallen away. I’m a situation. I’m holding my breath.“That’s if you want to, I mean, I don’t want to push my luck.” “I thought you didn’t make love. I thought you fucked hard.” I swallow, my mouth suddenly dry. He gives me a wicked grin, the effects of which travel all the way down there.“I can make an exception, or maybe combine the two, we’ll see. I really want to make love to you. Please, come to bed with me. I want our arrangement to work, but you re­ally need to have some idea what you’re getting yourself into. We can start your training tonight – with the basics. This doesn’t mean I’ve come over all hearts and flowers, it’s a means to an end, but one that I want, and hopefully you do too.” His gray gaze is intense.',  # int
  }

def find_book(features):



    piece = 'input_sample.txt'
    #piece ='./data/samples/fifty_shades.txt'
    piece_str = features['excerpt']
    print(piece_str)
    fout = open(piece,'w')
    fout.write(piece_str)


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
    corpus = pd.read_csv('./data/final_train.csv')
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


if __name__ == '__main__':
    print(find_book(example))
