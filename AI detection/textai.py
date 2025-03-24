import streamlit as st
import string
import pickle
from spacy.lang.en.stop_words import STOP_WORDS as stopwords

st.title('AI Detector')

input_text = st.text_area(label="Enter your Text")
run_button = st.button("Run")

contractions = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "could've": "could have", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it's": "it is", "let's": "let us", "mustn't": "must not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have",
    "wasn't": "was not", "won't": "would not", "would've": "would have", "wouldn't": "would not",
    " u ": " you ", " ur ": " your ", " n ": " and ", "dis": "this", "bak": "back", "brng": "bring"
}

def clean_text(text):
    tags = ['\n', '\'']
    for tag in tags:
        text = text.replace(tag, '')

    text = [x for x in text if x not in string.punctuation]
    text = ''.join(text)

    text = ' '.join([t for t in text.split() if t.lower() not in stopwords])

    if isinstance(text, str):
        for key, value in contractions.items():
            text = text.replace(key, value)

    return text
 

if run_button:
    if input_text.strip():  # Ensures input is not empty        
        cleaned_text = clean_text(input_text)
        
        # Load model only when the button is clicked
        with open('G:/Practice/SDP/AI detection/clf.pkl', 'rb') as clf_file, open('G:/Practice/SDP/AI detection/tfidf.pkl', 'rb') as tfidf_file:
            svm_clf = pickle.load(clf_file)
            svm_tfidf = pickle.load(tfidf_file)
        
        transformed_text = svm_tfidf.transform([cleaned_text])
        result = svm_clf.predict(transformed_text)

        if result[0]:
            st.markdown("Your text is AI-generated")
        else:
            st.markdown("Your text is human-written")
    else:
        st.warning("Please enter some text before running the detector.")
