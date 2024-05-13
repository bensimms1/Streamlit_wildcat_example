import streamlit as st
from fastai.vision.all import *
import altair as alt
import pandas as pd
import os
import urllib
import pathlib
from PIL import ImageOps
import pathlib

def main():
    st.title('CWS Course Wildcat Classifier')

    # Ensure all dependencies are downloaded
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)
    
    model = load_model()
    
    st.markdown("Upload a Scottish wildcat photo for classification.")
    image = st.file_uploader("", IMAGE_TYPES)
    if image:
        image_data = image.read()
        st.image(image_data, use_column_width=True)
        prediction = model.predict(image_data)
        pred_chart = predictions_to_chart(prediction, classes=model.dls.vocab)
        st.altair_chart(pred_chart, use_container_width=True)

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    # Check if the model file is downloaded and verified
    if not os.path.exists('model_1_wildcat.pkl') or (os.path.getsize('model_1_wildcat.pkl') != EXTERNAL_DEPENDENCIES['model_1_wildcat.pkl']['size']):
        download_file('model_1_wildcat.pkl')
    inf_model = load_learner('model_1_wildcat.pkl', cpu=True)
    return inf_model

@st.cache(show_spinner=False)
def download_file(file_path):
    # Don't download the file twice. If possible, verify the download using the file length.
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)
                    # Update the progress bar and warning message.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" % (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

def predictions_to_chart(prediction, classes):
    pred_rows = [{'class': classes[i], 'probability': round(float(conf) * 100, 2)} 
                 for i, conf in enumerate(prediction[2])]
    pred_df = pd.DataFrame(pred_rows).sort_values('probability', ascending=False).head(4)
    chart = alt.Chart(pred_df).mark_bar().encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=(0, 100))),
        y=alt.Y("class:N", sort=alt.EncodingSortField(field="probability", order="descending"))
    )
    return chart    

IMAGE_TYPES = ["png", "jpg"]

EXTERNAL_DEPENDENCIES = {
    "model_1_wildcat.pkl": {
        "url": "https://www.dropbox.com/scl/fi/o24f87s4kk3uv0el3vrg8/model_1_wildcat.pkl?rlkey=qc1qo7h67o5chdmp4vfppxd4d&st=5g65uwcn&dl=1",
        "size": 179191543
    }
}

if __name__ == "__main__":
    main()
