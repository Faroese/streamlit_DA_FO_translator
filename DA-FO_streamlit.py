# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:25:33 2022

@author: Heini
"""

#%% imports
import streamlit as st
from utils import translate_noatt

#%% Load model

# Hosted on my personal account until I figure something else out
#cloud_model_location = "1PmsUezmJGwTQP51yTMsjLGe2okdo-yxr"
cloud_model_location = "1jQZfAMXiTbzlBKxI2PuJPP-zRaLJwlGV"
@st.cache
def load_model():

    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    
    f_checkpoint = Path("model/skyAR_coord_resnet50.pt")

    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive(cloud_model_location, f_checkpoint)
    
    model = torch.load(f_checkpoint, map_location=device)
    model.eval()
    return model

#%% Streamlit
ex = ""
examples = ["Spiser du ikke til aften sammen med os?", "Jeg fik et nyt fotografiapparat"]

st.title("Dansk-Færøsk Translator!")
st.write("Eksempler på danske sætninger der kan oversættes")
for i in examples:
    ex += "- " + i + "\n"
st.markdown(ex)
input_text = st.text_input("Dansk \U0001F1E9\U0001F1F0")
output = translate_noatt(input_text)
st.text_area(label = "Færøsk \U0001F1EB\U0001F1F4 \U0001F970", value = output, height = 200, disabled = True)
