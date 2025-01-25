import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import gdown

# Application title
st.title("Seq2Seq Translation Application")
st.write("Use this application to translate texts using a Seq2Seq model.")

# Download and load the model from Google Drive
@st.cache_resource
def download_and_load_model():
    gdrive_link = "https://drive.google.com/uc?id=10jle6RnLUtLQEuFO2yoBefAogRJDYMjl"
    model_path = "seq2seq_model.h5"
    
    if not os.path.exists(model_path):
        try:
            st.write("Downloading the model...")
            gdown.download(gdrive_link, model_path, quiet=False)
        except Exception as e:
            st.error(f"Failed to download the model. Please check the link or your connection. Error: {e}")
            return None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load the model. Please ensure the file is a valid Keras model. Error: {e}")
        return None

# Load tokenizers
@st.cache_resource
def load_tokenizers():
    try:
        with open("src_tokenizer.pkl", "rb") as f:
            src_tokenizer = pickle.load(f)
        with open("tgt_tokenizer.pkl", "rb") as f:
            tgt_tokenizer = pickle.load(f)
        return src_tokenizer, tgt_tokenizer
    except FileNotFoundError as e:
        st.error(f"Tokenizer files are missing. Please ensure 'src_tokenizer.pkl' and 'tgt_tokenizer.pkl' exist. Error: {e}")
        return None, None

# Load the model and tokenizers
model = download_and_load_model()
src_tokenizer, tgt_tokenizer = load_tokenizers()

if model is not None and src_tokenizer is not None and tgt_tokenizer is not None:
    # Input text from the user
    input_text = st.text_input("Enter source text (the text you want to translate):", "")

    # Translate button
    if st.button("Translate"):
        if input_text:
            # Convert the source text to a sequence
            input_seq = pad_sequences(src_tokenizer.texts_to_sequences([input_text]), padding='post')

            # Get initial states from the model (Encoder States)
            encoder_model = model.get_layer("encoder")
            decoder_model = model.get_layer("decoder")

            state_h, state_c = encoder_model(input_seq)

            # Start translation with the <start> token
            tgt_seq = np.zeros((1, 1))
            tgt_seq[0, 0] = tgt_tokenizer.word_index.get("<start>", 0)

            translated_text = []

            for _ in range(20):  # Set a maximum translation length
                # Pass the current token to the decoder
                output, state_h, state_c = decoder_model(tgt_seq, state_h, state_c)
                predicted_id = tf.argmax(output[0, 0]).numpy()

                # Add the word to the translated text
                word = tgt_tokenizer.index_word.get(predicted_id, "")
                if word == "<end>":
                    break
                translated_text.append(word)

                # Update the current token
                tgt_seq = np.array([[predicted_id]])

            # Display the translation
            st.write("**Translated Text:**")
            st.write(" ".join(translated_text))
        else:
            st.write("Please enter the source text to translate.")
else:
    st.error("The application couldn't initialize because the model or tokenizers are missing.")
