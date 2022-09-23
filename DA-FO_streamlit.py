# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:25:33 2022

@author: Heini
"""

#%% imports
import streamlit as st
from utils import translate_noatt

#%% Streamlit
ex = ""
examples = ["Spiser du ikke til aften sammen med os?", "Jeg fik et nyt fotografiapparat"]

st.map([55.676098, 12.568337])
st.title("Dansk-Færøsk Translator!")
st.write("Eksempler på danske sætninger der kan oversættes")
for i in examples:
    ex += "- " + i + "\n"
st.markdown(ex)
input_text = st.text_input("Dansk \U0001F1E9\U0001F1F0")
output = translate_noatt(input_text)
st.text_area(label = "Færøsk \U0001F1EB\U0001F1F4 \U0001F970", value = output, height = 200, disabled = True)
