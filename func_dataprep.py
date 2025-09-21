# Import libraries
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import nltk, json 
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import string

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word.isalnum()
            and not word.isdigit()
            and word not in stop_words
            and word not in string.punctuation]
    return tokens

#Define a function that either shorten sentences or pads sentences with 0 to a fixed length
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

# Mapping function
def map_tokens(tokens, word2idx):
    return [word2idx.get(word, 0) for word in tokens]  # 0 if not in vocab

# Applied mapping function for 'tokens'
#pandas_data['indexed'] = pandas_data['input'].apply(lambda tokens: map_tokens(tokens, word2idx))