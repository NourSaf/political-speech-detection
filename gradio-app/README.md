
 👈🏽 👉🏽

# Political Speech Model 
This project implements a political speech analysis tool using a Doc2Vec model trained on German political speeches. The application allows users to input text and receive feedback on the political alignment of the speech, including the party affiliation, similarity score, and the most similar transcript.

## Model Information
This application uses a Doc2Vec model trained on German political speeches to classify text by political party affiliation.

## How to Use
1. Enter German political text in the input field
2. Click submit to analyze the text
3. View the predicted party alignment, confidence score, and similar transcript

## Technologies
- Gradio for the web interface
- Gensim Doc2Vec for document embedding and similarity
- NLTK for text processing

sdk: gradio <br>
sdk_version: 5.26.0<br>