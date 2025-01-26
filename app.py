import streamlit as st
import torch
import gdown
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# رابط Google Drive الخاص بالموديل
MODEL_URL = "https://drive.google.com/uc?id=Rcfe6q59rX6dnOc2kPTUN0nmIsKy3BB"

# تحميل النموذج
@st.cache_resource
def load_model():
    gdown.download(MODEL_URL, "model.pt", quiet=False)
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
    model.load_state_dict(torch.load("model.pt"))
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    return model, tokenizer

st.title("Seq2Seq Model Web App")
st.write("هذا التطبيق يستخدم النموذج لتحويل النصوص.")

# تحميل النموذج
model, tokenizer = load_model()

# واجهة المستخدم
input_text = st.text_area("أدخل النص هنا:", height=200)
if st.button("تحويل النص"):
    if input_text.strip():
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.text_area("الناتج:", result, height=200)
    else:
        st.warning("يرجى إدخال نص.")
