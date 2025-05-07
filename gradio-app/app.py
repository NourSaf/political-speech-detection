from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import nltk
nltk.download('punkt_tab')
import gradio as gr

# Load the training data
data = pd.read_json('data/training_data.json')

# Load the pre-trained model
model = Doc2Vec.load("model/political_doc2vec.model")

# Function to infer vector for a new piece of text
def infer_vector(model, text):
    tokenized = [x.lower() for x in nltk.word_tokenize(text, language='german')]
    vector = model.infer_vector(tokenized, epochs=20)
    return vector

# Function to get top similar speeches
def get_similar_speeches(model, input_text, topn=20):
    sample_vec = infer_vector(model, input_text)
    similar_docs = model.dv.most_similar([sample_vec], topn=topn)
    return similar_docs

# Function to calculate scores for each party and return the party with the highest score
def similar_docs_2_scores(similar_docs):
    scores = {}
    transcipt = ""
    
    for doc_id, similarity in similar_docs[:1]:
        num_id = int(doc_id.split("_")[1])
        transcipt = data.loc[num_id]["transcript"]

    for doc_id, similarity in similar_docs:
        num_id = int(doc_id.split("_")[1])
        label = data.loc[num_id]["labels"]
        scores[label] = scores.get(label, 0) + similarity

    doc_scores_list = [(k, v) for k, v in scores.items()]
    doc_highest_score = max(doc_scores_list, key=lambda x: x[1])
    party = doc_highest_score[0]
    similarity_score = (doc_highest_score[1] / 20) * 100

    full_score = sum(scores.values())
    afd_score = scores["AFD"]
    die_linke_score = scores["Die Linke"]
    right_confidence = round((afd_score / full_score) * 100, 2)
    left_confidence = round(die_linke_score / full_score * 100, 2)

    return party, similarity_score,right_confidence, left_confidence, transcipt

# Gradio interface function
def results(input_text):
    similar_docs = get_similar_speeches(model, input_text)
    party, similarity_score, right_confidence, left_confidence, transcript = similar_docs_2_scores(similar_docs)
    return party, similarity_score, right_confidence, left_confidence, transcript

# Setting up Gradio interface
iface = gr.Interface(fn=results, 
                     inputs="text", 
                     outputs=[
                         gr.Text(label="Political Party"),
                         gr.Number(label="Similarity Score"),
                         gr.Number(label="Right-Wing Confidence"),
                         gr.Number(label="Left-Wing Confidence"),
                         gr.Text(label="Similar Transcript")
                     ],
                     title="Political Speech Classifier",
                     description="Enter a political speech to classify its party affiliation and get the most similar transcript.")

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True)