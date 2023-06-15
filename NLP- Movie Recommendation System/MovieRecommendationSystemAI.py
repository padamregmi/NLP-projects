#!/usr/bin/env python
# coding: utf-8

# In[1]:


# BDM 3014 - Introduction to Artificial Intelligence -Project work

# project movie recommendation system

# project members

# Padam Bahadur Regmi(C0858265)
# Shreebatsa Aryal(C0859473)
# Bikesh Prajapati(C0859472)
# Hemanta Rijal(C0835075)
# Rosy Shrestha(C0857467)


# importing packages

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# In[2]:


import re

# In[3]:


# import movies data
file_movie = './tmdb_5000_movies.csv'
movies = pd.read_csv(file_movie)

# In[4]:


movies

# In[5]:


# import credits data
file_credit = "./tmdb_5000_credits.csv"
credits = pd.read_csv(file_credit)
credits

# In[6]:


# merging the two different datasets of movies and credits, and making the resultant "movies"
movies = movies.merge(credits, on='title')

# In[7]:


movies

# In[8]:


print(movies.info())

# In[9]:


# Out of the various different fields, we are considering to take following crucial fields for data analysis
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# In[10]:


movies

# In[11]:


# Finding out the Null values in particular fields
movies.isnull().sum()

# In[12]:


# Dropping the  rows where overview is NULL
movies.dropna(inplace=True)

# In[13]:


movies.duplicated().sum()

# In[14]:


import ast


# In[15]:


# Function to get the values of the key "name"
def convert(input):
    list = []
    for i in ast.literal_eval(input):
        list.append(i['name'])
    return list


# In[16]:


movies['genres'] = movies['genres'].apply(convert)

# In[17]:


movies['genres']

# In[18]:


movies.head()['genres']

# In[19]:


movies['keywords'] = movies['keywords'].apply(convert)

# In[20]:


movies['keywords']

# In[21]:


movies.head()['keywords']

# In[22]:


movies.head()['cast']


# In[23]:


# The Field 'cast' has different dictionary within it , which has information regarding the castField
# Converting string of list into list using literal_eval() function
def convertDict(castField):
    list = []
    counter = 0
    for i in ast.literal_eval(castField):
        while (counter) < 3:
            list.append(i['name'])
            counter += 1
        else:
            break
    return list


# In[24]:


movies['cast'] = movies['cast'].apply(convert)

# In[25]:


movies['cast']


# In[26]:

# From crew we will filter out name of the director of the movie
def get_director(crewDict):
    list = []
    for i in ast.literal_eval(crewDict):
        if i['job'] == 'Director':
            list.append(i['name'])
    return list


# In[27]:


movies['crew'] = movies['crew'].apply(get_director)

# In[28]:


movies['crew']

# In[29]:


# spliting the data and made a list of them i.e. space replaced by commas
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# In[30]:


movies['overview']


# In[31]:


# Function to remove spaces between values to treat as a single chunk to avoid the redundancy and making easy to analysis
def removeSpace(dataList):
    list = []
    for i in dataList:
        list.append(i.replace(" ", ""))
    return list


# In[32]:


movies['cast'] = movies['cast'].apply(removeSpace)
movies['keywords'] = movies['keywords'].apply(removeSpace)
movies['crew'] = movies['crew'].apply(removeSpace)
movies['genres'] = movies['genres'].apply(removeSpace)

# In[33]:


movies['cast']

# In[34]:


movies['keywords']

# In[35]:


movies['crew']

# In[36]:


movies['genres']

# In[37]:


# Adding all other salient fields to get a composite list called "tags"
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# In[38]:


movies['tags']

# In[39]:


new_movies = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])

# In[40]:


new_movies['tags'] = new_movies['tags'].apply(lambda x: " ".join(x))

# In[41]:


new_movies['tags'][0]

# In[42]:


new_movies['tags'] = new_movies['tags'].apply(lambda x: x.lower())

# In[43]:


new_movies['tags'][0]


# In[44]:


# handeling stem things in data
# for eg: "walk walking walked walks" are converted to "walk walk walk walk"

def stemFunc(tagValues):
    list = []
    for i in tagValues.split():
        list.append(ps.stem(i))
    return " ".join(list)


# In[45]:


new_movies['tags'] = new_movies['tags'].apply(stemFunc)

# In[46]:


from sklearn.feature_extraction.text import CountVectorizer

# Convert a collection of text documents to a matrix of token counts
# Removing the stop words
cv = CountVectorizer(max_features=6000, stop_words='english')

# Making arrays of vector of all the tags data i.e. converting text to a vector in 6000 Dimension
vector = cv.fit_transform(new_movies['tags']).toarray()
# Thus vector is of (4806, 6000) dimension


# In[47]:


from sklearn.metrics.pairwise import cosine_similarity

# Calculation of cosine similarities amongst vectors to calculate their distances to find the similarities among the movies
# It will calculate cosine distance (not Euclidean distance) of each movie to every other movies
similarVect = cosine_similarity(vector)


# In[48]:


def recommendMovies(search_movie):
    index_movie = new_movies[new_movies['title'] == search_movie].index[0]  # Fetching index of searched movie
    optimumDist = similarVect[index_movie]  # getting cosine value of that movie
    list_movies = sorted(list(enumerate(optimumDist)), reverse=True, key=lambda x: x[1])[1:6]
    # sorting the cosine values in descending order because highest value are most matched
    # enumerate function retain the index of the movies  even after sorting by creating tuple of (<index>,<cosine distance>)
    # lamda function used to sort on the basis of <cosine distance> rather than <index>

    for i in list_movies:
        print(new_movies.iloc[i[0]].title)


# In[66]:


# This function takes an input from user filters out the special characters
# calls the function recommendMovies() to provide suitable recommendations
def searchFunc():
    count = 0
    search = input('Enter a Movie Name: ')
    # search="Amazing-SpidermA"
    wordsList = search.split()
    searchOptimize = ""
    for j in wordsList:
        searchOptimize = ".*".join([searchOptimize, j])
    for i in new_movies['title']:
        x = re.search(re.sub("[$@&-:' ?,#_.!+(){}]", "", searchOptimize.upper()),
                      re.sub("[$@&-:' ?,#_.!+(){}]", "", i.upper()))  # Filters out the special characters
        if x:
            print('You Searched:', search)
            print('Search [', count + 1, ']=> Did You Mean?:', i)
            print('Recommended Movies are:')
            while (recommendMovies(i) != None): print(recommendMovies(i))
            print("***************************************************")
            count += 1
    if count == 0:
        print('The Movie ', search, ' is Not Available!!')


# In[67]:


searchFunc()

# In[ ]:




