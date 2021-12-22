import re
from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
import pymysql
from datetime import datetime
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
nltk.download('punkt')

# ===== from database to pandas ===== #
START_TIME = datetime.now()

# make a variable to connect with mysql database
conn = pymysql.connect(host="localhost",
                       user="root",
                       password='Dhqxlvmfkdla11@',
                       charset='utf8'
                       )
# cursor
cur = conn.cursor()

# select a database to use
cur.execute("USE rc_googlemovies") 

# sql query
sql = "SELECT * FROM contentsbased"
df = pd.read_sql_query(sql, conn)

END_TIME = datetime.now()
print(f"DATA LOADING TIME: {END_TIME - START_TIME}")

df['movie_id'] = df['movie_id'] - 1
print(df.head())

# ===== title_to_id ===== #
movie_title_to_id = {title:id for id, title in zip(df['movie_id'], df['title'])}
movie_id_to_title = {id:title for id, title in zip(df['movie_id'], df['title'])}

# ===== Word imbeddings ===== #

def preprocess(sentence):
    # lower()
    sentence = sentence.lower()

    # regular expression
    regex = r"[^a-zA-Z ]"
    subst = ""
    sentence = re.sub(regex, subst, sentence)

    return sentence

def remove_stop_words(sentence):
    # stopwords list
    stop_words = set(stopwords.words('english'))

    # tokenizing
    word_tokens = word_tokenize(sentence) 

    # remove stopwords
    for token in word_tokens:
        if token in stop_words:
            word_tokens.remove(token)
    
    return word_tokens

# 1) Preprocessing
df['title'] = [preprocess(t) for t in df['title']]
df['genre'] = [preprocess(g) for g in df['genre']]
df['overview'] = [preprocess(o) for o in df['overview']]

# 2) removing stopwords
df['tokens_t'] = [remove_stop_words(t) for t in df['title']]
df['tokens_g'] = [remove_stop_words(g) for g in df['genre']]
df['tokens_o'] = [remove_stop_words(o) for o in df['overview']]

# 3) Word2Vec
        # - tensorflow에서 pre-train된 word2vec 모델을 사용할 수 있지만 프로젝트는 직접 학습시켜 보기로 하자.
        # - fastext 는 colaborative filtering 프로젝트에서 진행해보자.
# CBOW : small data
EMBEDDING_SIZE = 50
WINDOW_SIZE = 2
embeddings_t = Word2Vec(sentences=df['tokens_t'], vector_size=EMBEDDING_SIZE, window=WINDOW_SIZE, min_count=1, workers=4, sg=0)
embeddings_g = Word2Vec(sentences=df['tokens_g'], vector_size=EMBEDDING_SIZE, window=WINDOW_SIZE, min_count=1, workers=4, sg=0)
embeddings_o = Word2Vec(sentences=df['tokens_o'], vector_size=EMBEDDING_SIZE, window=WINDOW_SIZE, min_count=1, workers=4, sg=0)

# embedding objects
embedding_t = embeddings_t.wv
embedding_g = embeddings_g.wv
embedding_o = embeddings_o.wv

# final embeddings
final_embedding_t = []
for tokens in df['tokens_t']:
    embedding = np.zeros(shape=(50,), dtype=np.float32)
    for token in tokens:
        embedding += embedding_t[token]
        embedding = embedding / len(df['tokens_t'])
    final_embedding_t.append(embedding)

final_embedding_g = []
for tokens in df['tokens_g']:
    embedding = np.zeros(shape=(50,), dtype=np.float32)
    for token in tokens:
        embedding += embedding_g[token]
        embedding = embedding / len(df['tokens_g'])
    final_embedding_g.append(embedding)

final_embedding_o = []
for tokens in df['tokens_o']:
    embedding = np.zeros(shape=(50,), dtype=np.float32)
    for token in tokens:
        embedding += embedding_o[token]
        embedding = embedding / len(df['tokens_o'])
    final_embedding_o.append(embedding)

# (final_embedding_t + final_embedding_g + final_embedding_o) / 3
final_embedding = []
for i in range(len(final_embedding_t)):
    avg = (final_embedding_t[i] + final_embedding_g[i] + final_embedding_g[i]) / 3
    final_embedding.append(avg)

# print(len(final_embedding))
# print(final_embedding)
# print("="*120)



# ===== Cosine Similarity ===== #

cos_sim = cosine_similarity(final_embedding, final_embedding)
print("\n")
print("*="*13,"Cosine similarity","*="*13)
print(cos_sim)

cos_sim_sorted_idx = cos_sim.argsort()[:, ::-1]
print("\n")
print("*="*8,"Similar movie index sorted by cosine sim","*="*7)
print(cos_sim_sorted_idx)



# ===== Find similar movies ===== #

def find_similar_movie(title, df, sorted_idx, top_n=5):
    
    # array : top 5 most similar movie 'ids'
    idx_recommended = sorted_idx[movie_title_to_id[title], 1:top_n+1]
        
    # dafaframe : top 5 most similar movies
    df_recommended = df[(df['movie_id'] == idx_recommended[0]) |      # 1st movie recommended
                        (df['movie_id'] == idx_recommended[1]) |      # 2nd movie recommended
                        (df['movie_id'] == idx_recommended[2]) |      # 3rd movie recommended
                        (df['movie_id'] == idx_recommended[3]) |      # 4th movie recommended
                        (df['movie_id'] == idx_recommended[4])]       # 5th movie recommended

    df_recommended = df_recommended[['title', 'genre', 'rate']]    

    return df_recommended



# ===== Execute ===== #

movies_recommended = find_similar_movie('Venom: Let There Be Carnage', df, cos_sim_sorted_idx)

print("\n")
print("*="*12,"YOUR NEXT MOVIES HERE!","*="*12)
print(movies_recommended)




# ===== Flask ===== #

app = Flask(__name__)
app.secret_key = "citizenyves"

@app.route("/hello")
def index():
    flash("A movie you've recently watched?")      
    value0 = movie_id_to_title[0]
    value1 = movie_id_to_title[1]
    value2 = movie_id_to_title[2]
    value3 = movie_id_to_title[3]
    value4 = movie_id_to_title[4]
    value5 = movie_id_to_title[5]
    value6 = movie_id_to_title[6]
    value7 = movie_id_to_title[7]
    value8 = movie_id_to_title[8]
    value9 = movie_id_to_title[9]
    value10 = movie_id_to_title[10]
    value11 = movie_id_to_title[11]
    value12 = movie_id_to_title[12]
    value13 = movie_id_to_title[13]
    value14 = movie_id_to_title[14]
    value15 = movie_id_to_title[15]
    value16 = movie_id_to_title[16]
    value17 = movie_id_to_title[17]
    value18 = movie_id_to_title[18]
    value19 = movie_id_to_title[19]
    value20 = movie_id_to_title[20]
    value21 = movie_id_to_title[21]
    value22 = movie_id_to_title[22]
    value23 = movie_id_to_title[23]
    value24 = movie_id_to_title[24]
    value25 = movie_id_to_title[25]
    value26 = movie_id_to_title[26]
    value27 = movie_id_to_title[27]
    value28 = movie_id_to_title[28]
    value29 = movie_id_to_title[29]
    value30 = movie_id_to_title[30]
    value31 = movie_id_to_title[31]
    value32 = movie_id_to_title[32]
    value33 = movie_id_to_title[33]
    value34 = movie_id_to_title[34]
    value35 = movie_id_to_title[35]
    value36 = movie_id_to_title[36]
    value37 = movie_id_to_title[37]
    value38 = movie_id_to_title[38]
    value39 = movie_id_to_title[39]
    value40 = movie_id_to_title[40]
    value41 = movie_id_to_title[41]
    value42 = movie_id_to_title[42]
    value43 = movie_id_to_title[43]
    value44 = movie_id_to_title[44]
    value45 = movie_id_to_title[45]
    value46 = movie_id_to_title[46]
    value47 = movie_id_to_title[47]
    value48 = movie_id_to_title[48]
    value49 = movie_id_to_title[49]
    value50 = movie_id_to_title[50]
    
    return render_template('index.html', value0=value0, 
    value1=value1, value2=value2, value3=value3, value4=value4, value5=value5, value6=value6, value7=value7, value8=value8, value9=value9, value10=value10,
    value11=value11, value12=value12, value13=value13, value14=value14, value15=value15, value16=value16, value17=value17, value18=value18, value19=value19, value20=value20,
    value21=value21, value22=value22, value23=value23, value24=value24, value25=value25, value26=value26, value27=value27, value28=value28, value29=value29, value30=value30,
    value31=value31, value32=value32, value33=value33, value34=value34, value35=value35, value36=value36, value37=value37, value38=value38, value39=value39, value40=value40,
    value41=value41, value42=value42, value43=value43, value44=value44, value45=value45, value46=value46, value47=value47, value48=value48, value49=value49, value50=value50) #html = python

@app.route("/greet", methods=["POST", "GET"])
def greet():
    flash('from "' + str(request.form['name_input']) + '" your next movies are..') #html 파일의 'name_input'과 매칭
    recommendation = find_similar_movie(movie_id_to_title[0], df, cos_sim_sorted_idx)
    return render_template("recommendation.html", recommendation=recommendation)


# if __name__ == '__main__':
#     app.run(debug=True)