import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model('seq2seq_model_advanced.h5')

# Load the Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([
    'hello', 'hi', 'how are you?', 'what is your name?', 'I am fine, thank you.',
    'good morning', 'good night', 'nice to meet you', 'how old are you?', 'where are you from?',
    'hola', '¿cómo estás?', '¿cuál es tu nombre?', 'estoy bien, gracias.', 'buenos días',
    'buenas noches', 'mucho gusto', '¿cuántos años tienes?', '¿de dónde eres?'
])

# Prediction function using the model
def predict_sequence(input_text):
    # Convert input text to sequence
    input_seq = tokenizer.texts_to_sequences([input_text])
    max_input_length = max([len(seq) for seq in input_seq])  # Find the max length for padding consistency
    input_seq = pad_sequences(input_seq, maxlen=max_input_length, padding='post')  # Ensure length consistency
    
    # Predict sequence
    predicted = model.predict(input_seq)
    predicted_index = np.argmax(predicted[0], axis=1)  # Get the index of the maximum probability
    
    # Convert predicted index to words (skip 0 as it represents padding)
    predicted_words = [tokenizer.index_word.get(i) for i in predicted_index if i != 0]

    # Check for words that the model can't translate
    untranslatable_words = []
    for word in input_text.split():
        if word.lower() not in tokenizer.word_index:
            untranslatable_words.append(word)
    
    # Construct the translated text
    predicted_text = ' '.join(predicted_words)
    
    # Add information about untranslatable words
    if untranslatable_words:
        untranslatable_message = f"Note: The following words were not translated: {' '.join(untranslatable_words)}"
    else:
        untranslatable_message = "All words were translated successfully."
    
    return predicted_text, untranslatable_message

# Streamlit UI
st.title("Simple Translation App")
input_text = st.text_input("Enter Text:")

if input_text:
    predicted_text, untranslatable_message = predict_sequence(input_text)
    st.write(f"Translated Text: {predicted_text}")
    st.write(untranslatable_message)



# Footer message
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); font-size: 14px; color: gray;">
        This page was created by <strong>Mohamed Hosam</strong>
    </div>
    """, unsafe_allow_html=True)
