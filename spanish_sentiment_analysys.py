# ! pip install spanish_sentiment_analysis
from classifier import * 
import streamlit

clf = SentimentClassifier()

x = "Esta muy buena esa pelicula"

st.success(x + ' ==> %.5f' % clf.predict(x))