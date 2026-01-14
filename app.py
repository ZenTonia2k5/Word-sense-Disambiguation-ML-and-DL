import streamlit as st
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ==========================================
# IMPORTS (Connecting to your files)
# ==========================================
# Make sure you renamed the files to: NaiveBayesWSD.py, KNNWSD.py, LSTMWSD.py
try:
    from NaiveBayesWSD import NaiveBayesWSD
    from KNNWSD import KNNWSD_Model
    from LSTMWSD import LSTMWSD, WSDDataset
except ImportError as e:
    st.error(f"Import Error: {e}. Did you rename the files to remove spaces?")
    st.stop()


# ==========================================
# UTILS (Data Cleaning)
# ==========================================
def clean_text(text):
    """Removes tags."""
    text = re.sub(r'\', '', text)
    return text.strip()


def load_data(word):
    try:
        with open(f"Sentence/{word}sentence.inp", "r", encoding="utf-8") as f:
            sentences = [clean_text(line) for line in f]
        with open(f"Sense/{word}sense.inp", "r", encoding="utf-8") as f:
            senses = [clean_text(line) for line in f]
        return sentences, senses
    except FileNotFoundError:
        return [], []


# ==========================================
# MODEL TRAINING HANDLERS
# ==========================================
@st.cache_resource
def get_naive_bayes(word):
    sentences, senses = load_data(word)
    if not sentences: return None

    # Pre-mark target
    marked_sentences = [s.replace(word, f"<target> {word} </target>") for s in sentences]

    # Initialize from your imported file
    model = NaiveBayesWSD()
    model.train(marked_sentences, senses)
    return model


@st.cache_resource
def get_knn(word):
    sentences, senses = load_data(word)
    if not sentences: return None

    marked_sentences = [s.replace(word, f"<target> {word} </target>") for s in sentences]

    # Initialize from your imported file
    model = KNNWSD_Model(k=5)
    model.train(marked_sentences, senses)
    return model


@st.cache_resource
def get_lstm(word):
    sentences, senses = load_data(word)
    if not sentences: return None

    marked_sentences = [s.replace(word, f"<target> {word} </target>") for s in sentences]

    # LSTM Setup (Reusing your WSDDataset logic)
    vocab = {"<pad>", "<unk>", "<target>", "</target>"}
    for s in marked_sentences:
        vocab.update(s.lower().split())

    word2idx = {w: i for i, w in enumerate(sorted(vocab))}
    sense_list = sorted(set(senses))
    sense2idx = {s: i for i, s in enumerate(sense_list)}
    idx2sense = {i: s for s, i in sense2idx.items()}

    # Use imported WSDDataset
    dataset = WSDDataset(marked_sentences, senses, word2idx, sense2idx)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize from your imported file
    device = torch.device("cpu")
    model = LSTMWSD(len(word2idx), 64, 64, len(sense2idx), word2idx["<pad>"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

    return model, word2idx, idx2sense, device


# ==========================================
# UI
# ==========================================
st.title("Word Sense Disambiguation System")
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Model", ["Naive Bayes", "kNN", "LSTM"])
target_word = st.sidebar.selectbox("Target Word", ["bank", "charge", "light", "match", "spring"])

user_input = st.text_area("Enter Sentence:", f"I went to the {target_word}.")

if st.button("Predict"):
    cleaned_input = clean_text(user_input)
    processed_input = cleaned_input.replace(target_word, f"<target> {target_word} </target>")

    prediction = "Error"

    if model_choice == "Naive Bayes":
        model = get_naive_bayes(target_word)
        if model: prediction = model.predict(processed_input)

    elif model_choice == "kNN":
        model = get_knn(target_word)
        if model: prediction = model.predict(processed_input)

    elif model_choice == "LSTM":
        lstm_data = get_lstm(target_word)
        if lstm_data:
            model, w2i, i2s, device = lstm_data
            model.eval()
            tokens = processed_input.lower().split()
            ids = [w2i.get(t, w2i["<unk>"]) for t in tokens]
            if len(ids) < 20:
                ids += [w2i["<pad>"]] * (20 - len(ids))
            else:
                ids = ids[:20]

            with torch.no_grad():
                pred_idx = model(torch.tensor([ids]).to(device)).argmax(1).item()
                prediction = i2s[pred_idx]

    st.success(f"Predicted Sense: **{prediction}**")