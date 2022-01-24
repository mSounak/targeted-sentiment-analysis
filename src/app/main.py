from fastapi import FastAPI, Query, APIRouter
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import pandas as pd
import uvicorn
from utils import preprocess, loop_pw
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(root_path='/api')

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the tokenizer
tokenizer = pickle.load(open('models/tokenizer.pickle', 'rb'))
# load the model
model = tf.keras.models.load_model('models/Tnet-LF-300dGLOVE')


class Data(BaseModel):
    """
    Data class to be used in the API
    """
    context: str
    target: str


@app.get("/healthz")
def healthz():
    """
    Health check
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(data: Data):
    """
    Predict the target value
    """
    recived = data.dict()

    context = recived['context']
    target = recived['target']
    
    sentence = preprocess(data.context)
    aspect = preprocess(data.target)

    # get position
    pw_t = loop_pw(sentence, aspect)
    
    seq_len = 46
    aspect_len = 8

    # tokenizer the sentence and target
    sen = tokenizer.texts_to_sequences([sentence])
    tar = tokenizer.texts_to_sequences([aspect])

    # pad the sentence
    sen = pad_sequences(sen, maxlen=seq_len, padding="post", truncating='post', value=0)
    tar = pad_sequences(tar, maxlen=aspect_len, padding="post", truncating='post', value=0)
    pw_t = pad_sequences([pw_t], maxlen=seq_len, padding="post", truncating='post', dtype='float64')
    # make prediction
    pred_prob = model.predict([sen, tar, pw_t])

    # predict the target
    pred = np.argmax(pred_prob, axis=1)

    probs = pred_prob[0].tolist()

    # Return the prediction
    return {
        "prediction": int(pred),
        "probability": list(probs)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)