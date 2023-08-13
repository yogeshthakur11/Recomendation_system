from flask import Flask, render_template, request
import pandas as pd
import joblib
from sqlalchemy import create_engine
import mysql.connector as connector

app = Flask(__name__)

con = connector.connect(host = 'localhost',
                        port = '3306',
                        user = 'root',
                        password = '12345',
                        database = 'recommenddb',
                        auth_plugin = 'mysql_native_password')

cur = con.cursor()
con.commit()

cur.execute('SELECT * FROM ent')
df = cur.fetchall()

ent = pd.DataFrame(df)
ent = ent.rename({0 : 'Id'}, axis = 1)
ent = ent.rename({1: 'Titles'}, axis = 1)
ent = ent.rename({2 : 'Category'}, axis = 1)
ent = ent.rename({3 : 'Reviews'}, axis = 1)


# ent = pd.read_csv("ent.csv", encoding = 'utf8')

#anime["genre"] = anime["genre"].fillna(" ")

tfidf = joblib.load('matrix')

tfidf_matrix = tfidf.transform(ent.Titles)

cosine_sim_matrix = joblib.load('cosine_matrix')

ent_idx = pd.Series(ent.index, index = ent['Titles']).drop_duplicates()

### Custom Function ###
def get_recommendations(Name, topN):
    #topN = 10
    #Getting the movie index using its title
    Title_id = ent_idx[Name]
    
    #getting the pairwise similarity score for all the titles with that title
    cosine_scores = list(enumerate(cosine_sim_matrix[Title_id]))
   
    #sorting the cosine_similarity based on scores
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    
    #get the scores of topN most similar movies
    cosine_scores_N = cosine_scores[0: topN + 1]
    
    #getting the movie index
    Title_idx =  [i[0] for i in cosine_scores_N]
    Title_scores = [i[1] for i in cosine_scores_N]
    
    #similar movies and scores
    
    Title_similar_show = pd.DataFrame(columns = ["name", "Score"])
    Title_similar_show["name"] = ent.loc[Title_idx, "Titles"]
    Title_similar_show["Score"] = Title_scores
    Title_similar_show.reset_index(inplace = True)
    #anime similar show.drop(["index"], axis = 1, inplace = True)
    return(Title_similar_show.iloc[1:, ])


######End of the Custom Function######    

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/guest', methods = ["post"])
def Guest():
    if request.method == 'POST' :
        mn = request.form["mn"]
        tp = request.form["tp"]
        
        q = get_recommendations(mn, topN = int(tp))
        # connection to a database
        
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",# user
                               pw = "12345",  # passwrd
                               db = "recommenddb")) # database
        
        # Transfering the file into a database by using the method "to_sql"
        q.to_sql('top_10', con = engine, if_exists = 'replace', chunksize = 1000, index= False)
        
        return render_template( "data.html", Y = "Results have been saved in your database", Z = q.to_html() )

if __name__ == '__main__':

    app.run(debug = True)