import streamlit as st
from fastai.vision.all import *
import altair as alt
import pandas as pd
import os
import urllib
import pathlib
from PIL import ImageOps

def main():
    st.title('CWS Course Wildcat classifier')
    model = load_model()  # Ensure model is loaded
    
    st.markdown("Upload a Scottish wildcat photo for classification.")
    image = st.file_uploader("", type=["png", "jpg"])
    if image:
        image_data = image.read()
        st.image(image_data, use_column_width=True)
        prediction = model.predict(image_data)
        pred_chart = predictions_to_chart(prediction, classes=model.dls.vocab)
        st.altair_chart(pred_chart, use_container_width=True)

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    if not os.path.exists('model_1_wildcat.pkl'):
        download_file('model_1_wildcat.pkl')
    inf_model = load_learner('model_1_wildcat.pkl', cpu=True)
    return inf_model

@st.cache(show_spinner=False)
def download_file(file_path):
    if os.path.exists(file_path):
        return
    with st.spinner(f"Downloading model..."):
        url = "https://www.dropbox.com/scl/fi/o24f87s4kk3uv0el3vrg8/model_1_wildcat.pkl?rlkey=qc1qo7h67o5chdmp4vfppxd4d&st=5g65uwcn&dl=1"
        urllib.request.urlretrieve(url, file_path)

def predictions_to_chart(prediction, classes):
    pred_rows = [{'class': classes[i], 'probability': round(float(conf) * 100, 2)} 
                 for i, conf in enumerate(prediction[2])]
    pred_df = pd.DataFrame(pred_rows).sort_values('probability', ascending=False).head(4)
    chart = alt.Chart(pred_df).mark_bar().encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=(0, 100))),
        y=alt.Y("class:N", sort=alt.EncodingSortField(field="probability", order="descending"))
    )
    return chart    

if __name__ == "__main__":
    main()
