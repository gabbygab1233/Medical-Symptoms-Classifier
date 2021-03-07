import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import spacy
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.feature_extraction.text import TfidfVectorizer

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key

model = load_model('model.pkl')

vec_model = load_model('vectorizer.pkl')

def main():
    st.beta_set_page_config(page_title="Medical Symptoms Text Classification", page_icon="💊", layout='centered', initial_sidebar_state='auto')
    # title
    html_temp = """
    <div>
    <h1 style="color:STEELBLUE;text-align:left;">Medical Symptoms Classifier  🩺 </h1>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    
    if st.checkbox("Information"):
        '''
        This web app uses machine learning to classify medical text according to the category of the ailment being described. 

        The model can classify 25  ailments ( **Emotional pain, Hair falling out, Head hurts, Infected wound, Foot achne, Shoulder pain,
        Injury from sports, Skin issue, Stomach ache, Knee pain, Joint pain, Hard to breath, Head ache, Body feels weak, Feeling dizzy, Back pain, 
        Open wounds, Internal pain, Blurry vision, Acne, Muscle pain, Neck pain, Cough, Ear ache, Feeling cold** ).
        '''
    '''
    ## How does it work ❓ 
    Write down how you feel and the machine learning model will classify the category of the ailment being described.
    '''
    st.image('demo.png')
    '''
    #### How are you feeling right now ? 
    '''
    med_text = st.text_area("", "Write Here")
    prediction_labels = {'Emotional pain': 0, 'Hair falling out':1, 'Head hurts':2, 'Infected wound':3, 'Foot achne':4,
    'Shoulder pain':5, 'Injury from sports':6, 'Skin issue':7, 'Stomach ache':8, 'Knee pain':9, 'Joint pain':10, 'Hard to breath':11,
    'Head ache':12, 'Body feels weak':13, 'Feeling dizzy':14, 'Back pain':15, 'Open wound':16, 'Internal pain':17, 'Blurry vision':18,
    'Acne':19, 'Neck pain':21, 'Cough':22, 'Ear achne':23, 'Feeling cold':24}
    
    if st.button("Classify"):
        vec_text =  vec_model.transform([med_text]).toarray()
        pred = model.predict(vec_text)
        final_result = get_key(pred,prediction_labels)
        st.warning((final_result))

    st.error("Note: This A.I application is for educational/demo purposes only and cannot be relied upon. Check the source code [here](https://github.com/gabbygab1233/Medical-Symptoms-Classifier)")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
