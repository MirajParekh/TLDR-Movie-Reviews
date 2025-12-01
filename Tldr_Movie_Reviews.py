"""
Utilizing sentiment analysis to determine whether
an inputted movie review is positive or negative

Purpose: to work with important libraries and gain a grasp of NLP
"""

__author__ = "Miraj Parekh"

import torch
import pandas as pd
from torch import nn
from transformers import AutoTokenizer

# 'tokenizer' is an instance of tokenizer class - provides ability to use methods 
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

# text = "this movie is lowkey the best movie i think i've ever seen, I cant lie"
# tokens = tokenizer(text)

# take data csv file and give raw data 
df = pd.read_csv("dataset.csv.zip")
print(df.head())

# convert sentiment column from strings to ints ('positive' - 1; 'negative' - 0)
def sentiment_conversion(word):
    if word == "positive":
        return 1
    return 0

df["sentiment"] = df["sentiment"].apply(sentiment_conversion) # <- letting pandas take care of it, thats why we exclude the parameter here bcus it wont work

