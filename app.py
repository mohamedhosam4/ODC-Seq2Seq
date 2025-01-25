
### Streamlit Code
import streamlit as st
import torch
import requests
from io import BytesIO

# Define the model loading and inference functions
def download_model_from_drive():
    url = "https://drive.google.com/uc?id=1XE1X39CzG2Ciiz55AxX-_rd7_85_eI7W"
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)

def load_model():
    encoder = Encoder(INPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(encoder, decoder, device).to(device)

    model_file = download_model_from_drive()
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    return model, device

def main():
    st.title("Seq2Seq Text Transformation")
    model, device = load_model()

    input_text = st.text_area("Enter input text:", "")
    if st.button("Generate Output"):
        if input_text:
            # Convert input_text to tensor
            src = torch.tensor([[int(x) for x in input_text.split()]], dtype=torch.long).to(device)
            trg = torch.zeros((10, 1), dtype=torch.long).to(device)  # Placeholder target tensor

            with torch.no_grad():
                output = model(src, trg, 0)  # No teacher forcing

            # Convert output tensor to text
            output_tokens = output.argmax(2).squeeze(1).tolist()
            output_text = " ".join(map(str, output_tokens))

            st.success(f"Generated Output: {output_text}")
        else:
            st.error("Please enter input text.")

if __name__ == "__main__":
    main()
