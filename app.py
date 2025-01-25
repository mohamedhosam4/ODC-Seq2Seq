
# 2. كود Streamlit لتحميل النموذج واستخدامه
# سيتم كتابة هذا الكود في Visual Studio Code أو أي بيئة محلية

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
import os

# تحميل النموذج (قم بتنزيله إذا لم يكن موجودًا محليًا)
MODEL_PATH = "seq2seq_model.h5"
GDRIVE_MODEL_URL = "https://drive.google.com/uc?id=1-NjnFP41zVpWvxvuY_jEcEsEarMkemnf"

if not os.path.exists(MODEL_PATH):
    gdown.download(GDRIVE_MODEL_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_seq2seq_model():
    return load_model(MODEL_PATH)

model = load_seq2seq_model()

# إعداد واجهة Streamlit
st.title("ترجمة باستخدام Seq2Seq")
st.write("أدخل النص باللغة الإنجليزية للحصول على الترجمة بالعربية.")
st.write("يمكنك إدخال الكلمات التالية للترجمة: hi, hello, how, good, morning, night, thanks, yes, no")

# إدخال النص
input_text = st.text_input("النص المدخل:")

# خريطة الأحرف (يجب مطابقتها مع النموذج المدرب)
input_characters = [' ', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'r', 's', 't', 'w', 'y']
target_characters = [' ', 'ا', 'ب', 'ح', 'د', 'ر', 'س', 'ش', 'ع', 'ف', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# تحويل النصوص إلى تمثيل رقمي
max_encoder_seq_length = 10  # حدد الطول الأقصى للتسلسل

def encode_input(input_text):
    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    for t, char in enumerate(input_text):
        if char in input_token_index:
            encoder_input_data[0, t, input_token_index[char]] = 1.
    return encoder_input_data

# توليد الترجمة
if input_text:
    encoder_input_data = encode_input(input_text.lower())
    decoder_input_data = np.zeros((1, 1, num_decoder_tokens))
    decoder_input_data[0, 0, target_token_index[' ']] = 1.0

    output_text = ""
    stop_condition = False
    while not stop_condition:
        decoder_output = model.predict([encoder_input_data, decoder_input_data])
        sampled_token_index = np.argmax(decoder_output[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        output_text += sampled_char

        if sampled_char == '\n' or len(output_text) > max_encoder_seq_length:
            stop_condition = True

        next_decoder_input = np.zeros((1, 1, num_decoder_tokens))
        next_decoder_input[0, 0, sampled_token_index] = 1.0
        decoder_input_data = next_decoder_input

    st.write("الترجمة:", output_text)