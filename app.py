import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# تحميل النموذج المدرب
model = tf.keras.models.load_model('seq2seq_model.h5')

# تحميل الـ Tokenizer (يجب أن تتأكد من تحميله بشكل صحيح، هنا استخدمنا نموذج تدريبي افتراضي)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(['hello', 'hi', 'how are you?', 'hola', '¿cómo estás?'])

# وظيفة التنبؤ باستخدام النموذج
def predict_sequence(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=5, padding='post')  # تأكد من تطابق الأطوال
    predicted = model.predict(input_seq)
    predicted_index = np.argmax(predicted[0], axis=1)
    return ' '.join([tokenizer.index_word.get(i) for i in predicted_index if i != 0])

# واجهة المستخدم باستخدام Streamlit
st.title("تطبيق Seq2Seq للمحادثات")
input_text = st.text_input("أدخل نصًا:")

if input_text:
    predicted_text = predict_sequence(input_text)
    st.write(f"النص المترجم: {predicted_text}")
