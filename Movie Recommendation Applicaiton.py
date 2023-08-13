#!/usr/bin/env python
# coding: utf-8

# **Problem Statement: The Media Streaming Giant** is an internet-based film streaming service aiming to enhance its movie catalog by spotlighting top-rated content and subsequently suggesting these films to users based on their viewing history. To facilitate this objective, the corporation has gathered the dataset and enlisted your expertise to offer analytical perspectives, as well as to devise a recommendation mechanism that streamlines the process of delivering targeted suggestions. The rating system spans from -9 to +9.

# In[1]:


# Let us start with importing the data on which we need to work and importing the libraries as well
import pandas as pd

movies = pd.read_csv(r"C:\Users\Yogesh Thakur\Downloads\Datasets_Recommendation Engine\Entertainment.csv")


# In[2]:


movies.shape


# In[3]:


movies.columns


# **Data Description: Entertainment Dataset**
# 
# ID -- Nominal ID of the movies
# 
# Titles -- Names of the movies
# 
# Category -- Category/ genre the film belonging to
# 
# Reviews -- Review rating of the movies by the users

# We happen to notice that the data has the names and category provided, which are in text format. We will have to decrypt the same using **TFIDF - "Term Frequency Inverse Document Frequency"** which will help us create a matrix of items and find the similarity matrix among the **Titles**.

# In[4]:


# Importing the TfidfVectorizer from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Creating TfidfVectorizer to remove all stop words

Tfidf = TfidfVectorizer(stop_words="english")


# In[5]:


# Checking for the NaN values in category
movies["Category"].isnull().sum()


# In[6]:


#creating tfidf matrix
tfidf_matrix = Tfidf.fit_transform(movies.Category)
tfidf_matrix.shape


# **Cosine Similarity**: Measures the cosine of the angle between two vectors. It is a judgment of orientation rather than magnitude between two vectors with respect to the origin. The cosine of 0 degrees is 1 which means the data points are similar and cosine of 90 degrees is 0 which means data points are dissimilar.

# In[7]:


# To find the similarity scores we import linear_kernel from sklearn
from sklearn.metrics.pairwise import linear_kernel


# In[8]:


# Creating Cosine similarity matrix, which will create the matrix of similarities 
# based on the magnitude calculated based on the cosine similarities

cos_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[9]:


# We now create a series of the movie titles, while removing the duplicate values
movies_index = pd.Series(movies.index, index = movies["Titles"]).drop_duplicates()


# In[10]:


movies_index


# In[11]:


# Checking the same for a random movie picked up
movies_id = movies_index["Heat (1995)"]
movies_id


# In[12]:


# We will have to create a user defined function for generating recommendations for the movies as under
def get_recommendations(Name, topN):
    
    # topN = 10
    # Getting the movie index using its title 
    movies_id = movies_index[Name]
    
    # Getting the pair wise similarity score for all the Titles using the cosine based similarities
    cosine_scores = list(enumerate(cos_sim_matrix[movies_id]))
    cosine_scores = sorted(cosine_scores, key= lambda x:x[1], reverse= True)
    
    # We get the scores of top N most similar movies
    cosine_scores_N = cosine_scores[0:topN+1]
    
    # Getting the movie index 
    movies_idx = [i[0] for i in cosine_scores_N]
    movies_scores = [i[1] for i in cosine_scores_N]
    
    movies_similar = pd.DataFrame(columns = ["Titles","Scores"])
    movies_similar["Titles"] = movies.loc[movies_idx, "Titles"]
    movies_similar["Scores"] = movies_scores
    movies_similar.reset_index(inplace = True)
    
    print(movies_similar)


# The above defined function helps us to recommend the movies based on the similarity on the categories they belong to. The scores are calculated for n number of similar movies and the recomendation for the similar movies is printed out. To understand better we write the code as below.

# In[14]:


# We are trying to recommend using the above defined function top 10 movies 
# that stand similar in category as that of the movie defined in the code

get_recommendations("To Die For (1995)", topN = 10) 
movies_index["To Die For (1995)"]


# In[ ]:


Hence, we see the result that clearly show the movies as above which match the closest to the movie defined above **"Casino (1995)"**

