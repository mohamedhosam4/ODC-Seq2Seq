import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model('seq2seq_model.h5')

# Load the Tokenizer (make sure it's properly loaded, here we use a simple training example)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(['hello', 'hi', 'how are you?', 'hola', '¿cómo estás?'])

# Prediction function using the model
def predict_sequence(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=5, padding='post')  # Ensure length consistency
    predicted = model.predict(input_seq)
    predicted_index = np.argmax(predicted[0], axis=1)
    return ' '.join([tokenizer.index_word.get(i) for i in predicted_index if i != 0])

# Streamlit UI
st.title("Seq2Seq Chat Application")
input_text = st.text_input("Enter Text:")

if input_text:
    predicted_text = predict_sequence(input_text)
    st.write(f"Translated Text: {predicted_text}")
