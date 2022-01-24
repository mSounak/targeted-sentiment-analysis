import pandas as pd
import numpy as np
import re
import string
import tensorflow as tf


def preprocess(text):
    # removing '\n', '\t'
    text = re.sub("\n", "", text)
    text = re.sub("\t", "", text)
    # lowercasing
    text = text.lower()

    # specific
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)

    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    
    # digits
    text = re.sub(r'\d+?\w', '', text)

    # Remove some punctuations 
    text = re.sub(r"[!?,'\"*)@#%(&$_.^-]", ' ', text)

    text = re.sub(' +', ' ', text)

    return text


def get_pw(k, m, i, n):
        C = 30.
        i += 1
        k += 1

        if i == k:
            pw = 1
        elif i < (k+m):
            pw = 1 - ((k + m - i)/C)
        elif (k + m) <= i and i <= n:
            pw = 1 - ((i - k)/C)
        else:
            pw = 0

        return round(pw, ndigits=3) if pw > 0 else 0

def loop_pw(sentence, target):

    text = sentence.split(" ")
    target_words = target.split(" ")

    first_target = [target_words[0]]
    m = len(first_target)
    n = len(text)

    if first_target[0] not in text:
        for i in range(len(text)):
            if target_words[0] in text[i]:
                k = i
                break

        pw = [get_pw(k, m, idx, n) for idx in range(len(text))]

    else:
        for i, word in enumerate(text):
            if target_words[0] in word:
                if len(target_words[0]) < len(word):
                    continue
                else:
                    k = i

        pw = [get_pw(k, m, idx, n) for idx in range(len(text))]

    return pw