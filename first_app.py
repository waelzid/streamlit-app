import streamlit as st  
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time
import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from wordcloud import WordCloud
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score ,classification_report, confusion_matrix , f1_score
import joblib 

from pylab import rcParams
import warnings
st.title("welcome to our project's website")
st.sidebar.text_input("insert the link of the store in yelp", key="URL")
st.sidebar.text_input("insert the link of the store in amazon", key="URL")
f=open("linearSVC.pkl",'rb')
mod1=pickle.load(f)
#x=mod1.predict(["this is a good place"])
#st.write(x)