import streamlit as st
import pandas as pd
from io import BytesIO, StringIO

file = st.file_uploader("Upload File", type=["png", "jpg"])
show_file = st.empty()
if not file:
            show_file.info("Please upload a file of type: " + ", ".join(["png", "jpg"]))
            return
        content = file.getvalue()
        if isinstance(file, BytesIO):
            show_file.image(file)
        file.close()

st.title("LesionNNet: Spinal Lesion Detection")
st.write("A 2 stage CNN Model that can detect spinal lesions, Please Upload your X-Ray Scan Image below")
st.write(" "
st.write("By Visharad Upadhyay")
