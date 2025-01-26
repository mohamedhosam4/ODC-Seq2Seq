import streamlit as st
import torch
import gdown
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_URL = "https://drive.google.com/uc?id=Rcfe6q59rX6dnOc2kPTUN0nmIsKy3BB"

@st.cache_resource
def load_model():
    gdown.download(MODEL_URL, "model.pt", quiet=False)
    if not os.path.exists("model.pt"):
        raise FileNotFoundError("الملف لم يتم تحميله. تحقق من الرابط أو الصلاحيات.")
    model = torch.load("model.pt")  # تحميل النموذج بالكامل
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    return model, tokenizer

st.title("Seq2Seq Model Web App")
st.write("هذا التطبيق يستخدم النموذج لتحويل النصوص.")

model, tokenizer = load_model()

input_text = st.text_area("أدخل النص هنا:", height=200)
if st.button("تحويل النص"):
    if input_text.strip():
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.text_area("الناتج:", result, height=200)
    else:
        st.warning("يرجى إدخال نص.")
