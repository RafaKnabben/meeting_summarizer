import streamlit as st
from pydub import AudioSegment
import requests

st.markdown('''## Video Audio Summarizer
''')

url = "https://test--2511-m67ja235na-ew.a.run.app/docs"


form = st.form(key="uploading")
url_input = form.text_input("Video link", "http://")

button = form.form_submit_button("Submit!")
button

params = {
    "url": url_input
    }

if button:
    res = requests.get(url, params=params)
    st.write(res.json())
