import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import gdown
import pickle

# Function to download the model from Google Drive using gdown
@st.cache_resource
def load_model_from_drive():
    # Google Drive model link (replace with actual file ID)
    gdrive_link = "https://drive.google.com/uc?id=10jle6RnLUtLQEuFO2yoBefAogRJDYMjl"

    # Download the model
    model_path = "seq2seq_model.h5"
    gdown.download(gdrive_link, model_path, quiet=False)
    
    # Load and return the model
    return tf.keras.models.load_model(model_path)

# Function to load tokenizer from Google Drive using gdown
@st.cache_resource
def load_tokenizer_from_drive(tokenizer_path):
    # Download the tokenizer
    gdown.download(tokenizer_path, tokenizer_path, quiet=False)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load the model
model = load_model_from_drive()

# Load tokenizers (replace these paths with the actual paths to your tokenizer files)
src_tokenizer = load_tokenizer_from_drive("src_tokenizer.pkl")
tgt_tokenizer = load_tokenizer_from_drive("tgt_tokenizer.pkl")

# Set up Streamlit page
st.title("Seq2Seq Model Translator")
st.write("Enter text for translation:")

# Define available languages
languages = {
    "English": ["Spanish", "French", "German"],
    "Spanish": ["English", "French"],
    "French": ["English", "Spanish"]
}

# User selects source and target languages
source_lang = st.selectbox("Select source language:", list(languages.keys()))
target_lang = st.selectbox("Select target language:", languages[source_lang])

# Input text from the user
input_text = st.text_input("Source text:")

# Display translation information
if input_text:
    st.write(f"Translating from {source_lang} to {target_lang}:")
    st.write(f"Source Text: {input_text}")

# Translation button
if st.button("Translate"):
    if input_text:
        # Convert source text to sequence
        input_seq = pad_sequences(src_tokenizer.texts_to_sequences([input_text]), padding='post')
        
        # Get encoder states
        state_h, state_c = model.encoder(input_seq)
        
        # Start the translation with the <start> token
        tgt_seq = np.zeros((1, 1))
        tgt_seq[0, 0] = tgt_tokenizer.word_index["<start>"]
        
        translated_text = []
        for _ in range(10):  # Limit translation length to 10 tokens
            # Get decoder output
            output, state_h, state_c = model.decoder(tgt_seq, state_h, state_c)
            predicted_id = tf.argmax(output[0, 0]).numpy()
            translated_text.append(tgt_tokenizer.index_word.get(predicted_id, ""))
            if tgt_tokenizer.index_word.get(predicted_id) == "<end>":
                break
            tgt_seq = np.array([[predicted_id]])
        
        # Show translated text
        st.write("Translated text: ", " ".join(translated_text))
    else:
        st.write("Please enter source text.")
