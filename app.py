import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import gdown

# Application title
st.title("Seq2Seq Translation Application")
st.write("Use this application to translate texts using a Seq2Seq model.")

# Define the custom Seq2Seq layer to avoid loading issues
class Seq2Seq(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

# Download and load the model from Google Drive
@st.cache_resource
def download_and_load_model():
    gdrive_link = "https://drive.google.com/uc?id=10jle6RnLUtLQEuFO2yoBefAogRJDYMjl"
    model_path = "seq2seq_model.h5"
    gdown.download(gdrive_link, model_path, quiet=False)
    with tf.keras.utils.custom_object_scope({'Seq2Seq': Seq2Seq}):
        model = tf.keras.models.load_model(model_path, compile=False)
    return model

# Load tokenizers
@st.cache_resource
def load_tokenizers():
    try:
        with open("src_tokenizer.pkl", "rb") as f:
            src_tokenizer = pickle.load(f)
        with open("tgt_tokenizer.pkl", "rb") as f:
            tgt_tokenizer = pickle.load(f)
        return src_tokenizer, tgt_tokenizer
    except FileNotFoundError:
        st.error("Tokenizer files are missing. Please ensure 'src_tokenizer.pkl' and 'tgt_tokenizer.pkl' are available.")
        return None, None

# Load the model and tokenizers
model = download_and_load_model()
src_tokenizer, tgt_tokenizer = load_tokenizers()

if model is None or src_tokenizer is None or tgt_tokenizer is None:
    st.error("The application couldn't initialize because the model or tokenizers are missing.")
else:
    # Input text from the user
    input_text = st.text_input("Enter source text (the text you want to translate):", "")

    # Translate button
    if st.button("Translate"):
        if input_text:
            # Convert the source text to a sequence
            input_seq = pad_sequences(src_tokenizer.texts_to_sequences([input_text]), padding='post')

            # Get encoder states
            encoder_model = model.encoder
            decoder_model = model.decoder

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
