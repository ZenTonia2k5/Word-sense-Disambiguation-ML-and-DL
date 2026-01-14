import streamlit as st
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==========================================
# IMPORTS (Connecting to your files)
# ==========================================
# Make sure you renamed the files to: NaiveBayesWSD.py, KNNWSD.py, LSTMWSD.py
try:
    from NaivesBayes import NaiveBayesWSD
    from kNN import KNNWSD
    from LSTM import LSTMWSD, WSDDataset
except ImportError as e:
    st.error(f"Import Error: {e}. Did you rename the files to remove spaces?")
    st.stop()


# ==========================================
# UTILS (Data Cleaning)
# ==========================================
def clean_text(text):
    """Removes tags."""
    text = re.sub(r'\'', '', text)
    return text.strip()


def load_data(word):
    try:
        with open(f"Sentence/{word}sentence.inp", "r", encoding="utf-8") as f:
            sentences = [clean_text(line) for line in f]
        with open(f"Sense/{word}sense.inp", "r", encoding="utf-8") as f:
            senses = [clean_text(line) for line in f]
        return sentences, senses
    except FileNotFoundError:
        st.error(f"Files for '{word}' not found.")
        return [], []


def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, precision, recall, f1


# ==========================================
# MODEL TRAINING HANDLERS (Now with Eval)
# ==========================================
@st.cache_resource
def get_naive_bayes(word):
    sentences, senses = load_data(word)
    if not sentences: return None, None

    # Pre-mark target
    marked_sentences = [s.replace(word, f"<target> {word} </target>") for s in sentences]

    # Split Data 80/20
    data = list(zip(marked_sentences, senses))
    random.seed(42)
    random.shuffle(data)
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]

    train_sent = [x[0] for x in train_data]
    train_lbl = [x[1] for x in train_data]
    test_sent = [x[0] for x in test_data]
    test_lbl = [x[1] for x in test_data]

    # Train
    model = NaiveBayesWSD()
    model.train(train_sent, train_lbl)

    # Evaluate
    y_pred = [model.predict(s) for s in test_sent]
    metrics = calculate_metrics(test_lbl, y_pred)

    return model, metrics


@st.cache_resource
def get_knn(word):
    sentences, senses = load_data(word)
    if not sentences: return None, None

    marked_sentences = [s.replace(word, f"<target> {word} </target>") for s in sentences]

    # Split Data 80/20
    data = list(zip(marked_sentences, senses))
    random.seed(42)
    random.shuffle(data)
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]

    train_sent = [x[0] for x in train_data]
    train_lbl = [x[1] for x in train_data]
    test_sent = [x[0] for x in test_data]
    test_lbl = [x[1] for x in test_data]

    # Train
    model = KNNWSD(k=5)
    model.train(train_sent, train_lbl)

    # Evaluate
    y_pred = [model.predict(s) for s in test_sent]
    metrics = calculate_metrics(test_lbl, y_pred)

    return model, metrics


@st.cache_resource
def get_lstm(word):
    sentences, senses = load_data(word)
    if not sentences: return None, None, None, None, None

    marked_sentences = [s.replace(word, f"<target> {word} </target>") for s in sentences]

    # Vocab & Indexing
    vocab = {"<pad>", "<unk>", "<target>", "</target>"}
    for s in marked_sentences:
        vocab.update(s.lower().split())

    word2idx = {w: i for i, w in enumerate(sorted(vocab))}
    sense_list = sorted(set(senses))
    sense2idx = {s: i for i, s in enumerate(sense_list)}
    idx2sense = {i: s for s, i in sense2idx.items()}

    # Split Data 80/20
    data = list(zip(marked_sentences, senses))
    random.seed(42)
    random.shuffle(data)
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]

    # Datasets
    train_dataset = WSDDataset([x[0] for x in train_data], [x[1] for x in train_data], word2idx, sense2idx)
    test_dataset = WSDDataset([x[0] for x in test_data], [x[1] for x in test_data], word2idx, sense2idx)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)

    # Initialize Model
    device = torch.device("cpu")
    model = LSTMWSD(len(word2idx), 64, 64, len(sense2idx), word2idx["<pad>"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Train Loop
    model.train()
    for epoch in range(10):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

    # Evaluate Loop
    model.eval()
    y_true_eval = []
    y_pred_eval = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().tolist()
            y_pred_eval.extend(preds)
            y_true_eval.extend(y.tolist())

    # Convert indices back to sense strings for consistency if needed,
    # but metrics calc works fine with indices too.
    metrics = calculate_metrics(y_true_eval, y_pred_eval)

    return model, word2idx, idx2sense, device, metrics


# ==========================================
# UI
# ==========================================
st.title("Word Sense Disambiguation System")

# --- SIDEBAR CONFIG ---
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Model", ["Naive Bayes", "kNN", "LSTM"])
target_word = st.sidebar.selectbox("Target Word", ["bank", "charge", "light", "match", "spring"])

# --- DISPLAY METRICS IN SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")
metrics = None

# Train/Load model based on selection
if model_choice == "Naive Bayes":
    with st.spinner("Training Naive Bayes..."):
        nb_model, metrics = get_naive_bayes(target_word)
        model = nb_model

elif model_choice == "kNN":
    with st.spinner("Training kNN..."):
        knn_model, metrics = get_knn(target_word)
        model = knn_model

elif model_choice == "LSTM":
    with st.spinner("Training LSTM..."):
        lstm_out = get_lstm(target_word)
        if lstm_out[0] is not None:
            model, w2i, i2s, device, metrics = lstm_out
        else:
            model = None

# Show Metrics if available
if metrics:
    acc, prec, rec, f1 = metrics
    st.sidebar.metric("Accuracy", f"{acc:.2%}")
    st.sidebar.metric("Precision", f"{prec:.3f}")
    st.sidebar.metric("Recall", f"{rec:.3f}")
    st.sidebar.metric("F1 Score", f"{f1:.3f}")
else:
    st.sidebar.warning("Could not train model.")

# --- MAIN PREDICTION UI ---
st.markdown(f"### Predict Sense for '*{target_word}*'")
user_input = st.text_area("Enter Sentence:", f"I went to the {target_word}...")

if st.button("Predict"):
    cleaned_input = clean_text(user_input)
    processed_input = cleaned_input.replace(target_word, f"<target> {target_word} </target>")

    prediction = "Error"

    if not model:
        st.error("Model not loaded.")
    else:
        try:
            if model_choice == "Naive Bayes":
                prediction = model.predict(processed_input)

            elif model_choice == "kNN":
                prediction = model.predict(processed_input)

            elif model_choice == "LSTM":
                # LSTM inference logic
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

        except Exception as e:
            st.error(f"Prediction Error: {e}")