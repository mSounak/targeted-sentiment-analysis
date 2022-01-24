import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import preprocess, loop_pw
import requests
import pandas as pd
import altair as alt
import json

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)



def predict(input_text, aspect):
    
    data = {"context": input_text, "target": aspect}
    res = requests.post(
        'http://192.168.49.2/api/predict', json=data)

    rest = res.json()
    return rest['prediction'], rest['probability']


# Main Body
st.title("Aspect-based Sentiment Analysis")

input_text = st.text_area("Enter your text", value="")
input_aspect = st.text_input("Enter your aspect", value="")
p_text = preprocess(text=input_text)
p_aspect = preprocess(text=input_aspect)

flag = st.button("Predict")

if flag:
    if p_aspect not in p_text:
        st.error("Aspect not in text")
    else:
        pred, probs = predict(p_text, p_aspect)

        if pred == 0:
            st.error(f"Negative | Confidence: {max(probs)*100:.2f}")
        elif pred == 1:
            st.warning(f"Neutral | Confidence: {max(probs)*100:.2f}")
        elif pred == 2:
            st.success(f"Positive | Confidence: {max(probs)*100:.2f}")

        col = ['Negative', 'Neutral', 'Positive']
        
        prob = [i * 100 for i in probs]
        df = pd.DataFrame({
            'sentiment': col,
            'probability': prob,
            'color': ['#ff4747', '#ffcb47', '#2bff56']
        }).sort_values('probability', ascending=False)
        
        st.write(alt.Chart(df).mark_bar(size=50).encode(
            x='probability',
            y='sentiment',
            color=alt.Color("color", scale=None)
        ).properties(width=500, height=300))

        


        


        