# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:25:33 2022

@author: Heini
"""

#%% imports
import streamlit as st
import pandas as pd
from utils import translate_noatt

#%% Streamlit
ex = ""
examples = ["Spiser du ikke til aften sammen med os?", "Jeg fik et nyt fotografiapparat.", "Jeg tror de så os.", "Jeg læste en bog."]

df = pd.DataFrame([[55.676098, 12.568337], [62.007864, -6.790982]], columns = ["lat", "lon"])
                  
st.title("Dansk-Færøsk Translator \U0001F1E9\U0001F1F0 \U0001F1EB\U0001F1F4")
#st.write("Eksempler på danske sætninger der kan oversættes")
for i in examples:
    ex += "- " + i + "\n"
st.markdown("Eksempler på danske sætninger der kan oversættes" + "\n" + ex)
input_text = st.text_input("Dansk \U0001F1E9\U0001F1F0")
output = translate_noatt(input_text)
st.text_area(label = "Færøsk \U0001F1EB\U0001F1F4", value = output, height = 200, disabled = True)
st.map(df)
