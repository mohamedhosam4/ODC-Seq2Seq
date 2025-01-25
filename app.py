import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
import os

# رابط Google Drive لتحميل النموذج
MODEL_PATH = "simple_seq2seq_model.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=1-Rcfe6q59rX6dnOc2kPTUN0nmIsKy3BB"

# التحقق من وجود النموذج، وإذا لم يكن موجودًا، يتم تنزيله
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# تحميل النموذج باستخدام Streamlit Cache
@st.cache_resource
def load_seq2seq_model():
    return load_model(MODEL_PATH)

model = load_seq2seq_model()

# خريطة الأحرف (يجب أن تطابق التدريب)
input_characters = ['e', 'h', 'i', 'k', 'l', 'n', 'o', 's', 't', 'y']
target_characters = ['ا', 'ب', 'ح', 'د', 'ر', 'س', 'ش', 'ع', 'ف', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

input_token_index = {char: i for i, char in enumerate(input_characters)}
target_token_index = {char: i for i, char in enumerate(target_characters)}
reverse_target_char_index = {i: char for char, i in target_token_index.items()}

# تحويل النصوص إلى بيانات يمكن معالجتها بالنموذج
def encode_input_text(text):
    encoder_input_data = np.zeros((1, len(text), num_encoder_tokens), dtype="float32")
    for t, char in enumerate(text):
        if char in input_token_index:
            encoder_input_data[0, t, input_token_index[char]] = 1.0
    return encoder_input_data

# توليد الترجمة باستخدام النموذج
def decode_sequence(input_seq):
    states_value = model.predict([input_seq, np.zeros((1, 1, num_decoder_tokens))])[1:]
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['ا']] = 1.0

    decoded_sentence = ''
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = model.predict([input_seq, target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > 10:
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [h, c]

    return decoded_sentence

# واجهة المستخدم مع Streamlit
st.title("تطبيق الترجمة باستخدام Seq2Seq")
st.write("أدخل كلمة أو نصًا باللغة الإنجليزية للحصول على الترجمة بالعربية.")

# إدخال المستخدم
input_text = st.text_input("النص المدخل:")

if input_text:
    input_seq = encode_input_text(input_text.lower())
    translation = decode_sequence(input_seq)
    st.write("الترجمة:", translation)
