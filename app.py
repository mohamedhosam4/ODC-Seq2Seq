import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import gdown

# Set application title
st.title("Seq2Seq Translation Application")
st.write("Use this application to translate texts using a Seq2Seq model.")

# Function to download and load the model
@st.cache_resource
def download_and_load_model():
    # Google Drive link for the pre-trained model
    gdrive_link = "https://drive.google.com/uc?id=1lwZe4DcfQZSKsmoZCArW2qm2C6pUW_eO"
    model_path = "seq2seq_model.h5"
    gdown.download(gdrive_link, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    return model

# Function to load the tokenizers
@st.cache_resource
def load_tokenizers():
    # Google Drive links for the tokenizers
    src_tokenizer_link = "https://drive.google.com/uc?id=103LXD0rPAI9qZNsWx1zaBw1lL0UjhZks"
    tgt_tokenizer_link = "https://drive.google.com/uc?id=1p2yZ-_IGMn6np_yHgmRv3PvPZtcz2ERS"

    # Download tokenizers
    gdown.download(src_tokenizer_link, "src_tokenizer.pkl", quiet=False)
    gdown.download(tgt_tokenizer_link, "tgt_tokenizer.pkl", quiet=False)
    
    with open("src_tokenizer.pkl", "rb") as f:
        src_tokenizer = pickle.load(f)
    with open("tgt_tokenizer.pkl", "rb") as f:
        tgt_tokenizer = pickle.load(f)
    return src_tokenizer, tgt_tokenizer

# Load the model and tokenizers
model = download_and_load_model()
src_tokenizer, tgt_tokenizer = load_tokenizers()

# Check if resources are successfully loaded
if model and src_tokenizer and tgt_tokenizer:
    # Input field for user text
    input_text = st.text_input("Enter the text you want to translate:", "")

    if st.button("Translate"):
        if input_text:
            # Preprocess input text
            input_seq = pad_sequences(
                src_tokenizer.texts_to_sequences([input_text]), padding="post"
            )
            
            # Perform translation
            output_seq = model.predict(input_seq)
            translated_text = " ".join(
                [tgt_tokenizer.index_word.get(idx, "") for idx in np.argmax(output_seq, axis=-1)[0]]
            )
            
            # Display the translated text
            st.write("**Translated Text:**")
            st.write(translated_text)
        else:
            st.error("Please enter some text to translate.")
else:
    st.error("The model or tokenizers could not be loaded. Please ensure they are correctly uploaded.")
