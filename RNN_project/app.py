import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

word_index=imdb.get_word_index()
revers_word_index={value:key for key,value in word_index.items()}


model=load_model('simple_rnn_imdb.h5')


def preprocessing_text(text):
    words=text.lower().split()
    encode_Review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encode_Review],maxlen=500)
    return padded_review



def predict_dentiment(review):
    preprocessed_input=preprocessing_text(review)
    
    prediction=model.predict(preprocessed_input)
    
    sentiment='Positive' if prediction[0][0] > 0.5 else 'negative'
    
    return sentiment,prediction[0][0]



st.title('IMDB MOVIE REVIEW ANALAYISIS')
st.write('This is a simple IMDB movie review sentiment analysis using RNN')
user_input=st.text_area('Enter your review here')

if st.button('Classify'):
    preprocess_input=preprocessing_text(user_input)
    
    preddiciton=model.predict(preprocess_input)
    
    
    sentiment='Positive' if preddiciton[0][0] > 0.5 else 'negative'
    st.write(f'Sentiment {sentiment}')
    st.write(f'prediction score {preddiciton[0][0]*100}')

else:
    st.write('please neter movei review')    