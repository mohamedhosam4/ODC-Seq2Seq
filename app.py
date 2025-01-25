import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import gdown

# Application Title
st.title("Seq2Seq Translation Application")
st.write("Translate text using a Seq2Seq model trained on your dataset.")

# Function to download and load the Seq2Seq model
@st.cache_resource
def download_and_load_model():
    try:
        # Google Drive link for the model
        gdrive_link = "https://drive.google.com/uc?id=1lwZe4DcfQZSKsmoZCArW2qm2C6pUW_eO"
        model_path = "seq2seq_model.h5"

        # Download the model
        gdown.download(gdrive_link, model_path, quiet=False)

        # Load the model
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to download and load the tokenizers
@st.cache_resource
def load_tokenizers():
    try:
        # Google Drive links for the tokenizers
        src_tokenizer_link = "https://drive.google.com/uc?id=103LXD0rPAI9qZNsWx1zaBw1lL0UjhZks"
        tgt_tokenizer_link = "https://drive.google.com/uc?id=1p2yZ-_IGMn6np_yHgmRv3PvPZtcz2ERS"

        # Paths to save tokenizers locally
        src_tokenizer_path = "src_tokenizer.pkl"
        tgt_tokenizer_path = "tgt_tokenizer.pkl"

        # Download the tokenizers
        gdown.download(src_tokenizer_link, src_tokenizer_path, quiet=False)
        gdown.download(tgt_tokenizer_link, tgt_tokenizer_path, quiet=False)

        # Load tokenizers from the downloaded files
        with open(src_tokenizer_path, "rb") as src_file:
            src_tokenizer = pickle.load(src_file)

        with open(tgt_tokenizer_path, "rb") as tgt_file:
            tgt_tokenizer = pickle.load(tgt_file)

        return src_tokenizer, tgt_tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizers: {e}")
        return None, None

# Load the model and tokenizers
model = download_and_load_model()
src_tokenizer, tgt_tokenizer = load_tokenizers()

if model is None or src_tokenizer is None or tgt_tokenizer is None:
    st.error("Failed to load the model or tokenizers. Please check the links and files.")
else:
    # Input text from the user
    input_text = st.text_input("Enter text to translate:")

    # Translate button
    if st.button("Translate"):
        if input_text:
            try:
                # Convert the input text to a sequence
                input_seq = pad_sequences(src_tokenizer.texts_to_sequences([input_text]), maxlen=20, padding='post')

                # Perform translation using the model
                encoder_model = model.layers[0]
                decoder_model = model.layers[1]

                state_h, state_c = encoder_model.predict(input_seq)
                tgt_seq = np.zeros((1, 1))
                tgt_seq[0, 0] = tgt_tokenizer.word_index.get('<start>', 0)

                translated_text = []
                for _ in range(20):  # Limit the translation to 20 words
                    output_tokens, state_h, state_c = decoder_model.predict([tgt_seq, state_h, state_c])
                    predicted_id = np.argmax(output_tokens[0, -1, :])

                    predicted_word = tgt_tokenizer.index_word.get(predicted_id, '')
                    if predicted_word == '<end>':
                        break
                    translated_text.append(predicted_word)

                    # Update target sequence with the predicted ID
                    tgt_seq = np.array([[predicted_id]])

                # Display the translated text
                st.write("**Translated Text:**")
                st.write(" ".join(translated_text))
            except Exception as e:
                st.error(f"Translation error: {e}")
        else:
            st.warning("Please enter text to translate.")
