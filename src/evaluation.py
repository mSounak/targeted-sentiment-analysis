import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from preprocess import preprocess, loop_pw
import pickle as pkl
from tqdm import tqdm

model = load_model('models/Tnet-LF-300dGLOVE')
tokenizer = pkl.load(open('models/tokenizer.pickle', 'rb'))


data = pd.read_csv('data/test.csv')

data['clean_text'] = data['text'].apply(preprocess)
data['clean_aspect'] = data['aspect'].apply(preprocess)

seq_len = 46
aspect_len = 8
test_label = []
for i in tqdm(range(len(data))):
    sentence, target = data['clean_text'][i], data['clean_aspect'][i]
    pw_t = loop_pw(sentence, target)
    sen = tokenizer.texts_to_sequences([sentence])
    tar = tokenizer.texts_to_sequences([target])

    sen = pad_sequences(sen, maxlen=seq_len, padding="post", truncating='post', value=0)
    tar = pad_sequences(tar, maxlen=aspect_len, padding="post", truncating='post', value=0)
    pw_t = pad_sequences([pw_t], maxlen=seq_len, padding="post", truncating='post', dtype='float64')

    probs = model.predict([sen, tar, pw_t])
    test_label.append(np.argmax(probs, axis=1)[0])

result = pd.DataFrame({
    'text' : data['text'],
    'aspect' : data['aspect'],
    'label' : test_label
})

result.to_csv('data/results/test.csv', index=False)