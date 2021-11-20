import streamlit as st
import os
'''
# Speech Summarizer
'''

st.markdown('''Our application allows users to create a summary of a given speech on video or audio.
            Simply use the features below and it will output an abstractive summary from your given media input.

''')

"""## Upload your file: """

filename = st.text_input('Enter a file path:')
try:
    with open(filename) as input:
        st.text(input.read())
except FileNotFoundError:
    st.error('File not found.')

'''
## Our model will now work create a summary transcript of your media input.

'''

"""
## Here is your summary
"""
