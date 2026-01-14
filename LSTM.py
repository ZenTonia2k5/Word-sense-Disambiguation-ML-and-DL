import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =============================
# Dataset
# =============================
class WSDDataset(Dataset):
    def __init__(self, sentences, senses, word2idx, sense2idx, max_len=20):
        self.sentences = sentences
        self.labels = [sense2idx[s] for s in senses]
        self.word2idx = word2idx
        self.max_len = max_len

    def encode(self, sentence):
        tokens = sentence.lower().split()
        ids = [self.word2idx.get(t, self.word2idx["<unk>"]) for t in tokens]

        if len(ids) < self.max_len:
            ids += [self.word2idx["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids)

    def __getitem__(self, idx):
        return self.encode(self.sentences[idx]), torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.sentences)

# =============================
# LSTM Model (BiLSTM)
# =============================
class LSTMWSD(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)   # sentence pooling
        return self.fc(out)

# =============================
# Training
# =============================
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

# =============================
# Evaluation with Metrics
# =============================
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.tolist())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return acc, precision, recall, f1

# =============================
# Predict multiple sentences
# =============================
def predict_sentences(model, sentences, word2idx, idx2sense, max_len, device):
    model.eval()
    results = []

    with torch.no_grad():
        for s in sentences:
            tokens = s.lower().split()
            ids = [word2idx.get(t, word2idx["<unk>"]) for t in tokens]

            if len(ids) < max_len:
                ids += [word2idx["<pad>"]] * (max_len - len(ids))
            else:
                ids = ids[:max_len]

            x = torch.tensor([ids]).to(device)
            pred = model(x).argmax(1).item()
            results.append((s, idx2sense[pred]))

    return results

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    word = input("Enter target word: ").strip().lower()

    with open(f"Sentence/{word}sentence.inp", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f]

    with open(f"Sense/{word}sense.inp", "r", encoding="utf-8") as f:
        senses = [line.strip() for line in f]

    assert len(sentences) == len(senses)

    # Target word marking
    sentences = [
        s.replace(word, f"<target> {word} </target>")
        for s in sentences
    ]

    # Vocabulary
    vocab = {"<pad>", "<unk>", "<target>", "</target>"}
    for s in sentences:
        vocab.update(s.lower().split())

    word2idx = {w: i for i, w in enumerate(sorted(vocab))}
    pad_idx = word2idx["<pad>"]

    sense_list = sorted(set(senses))
    sense2idx = {s: i for i, s in enumerate(sense_list)}
    idx2sense = {i: s for s, i in sense2idx.items()}

    dataset = WSDDataset(sentences, senses, word2idx, sense2idx)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMWSD(
        vocab_size=len(word2idx),
        emb_dim=64,
        hidden_dim=64,
        num_classes=len(sense2idx),
        pad_idx=pad_idx
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print("\nTraining LSTM...")
    for epoch in range(10):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        acc, p, r, f1 = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch+1}: "
            f"Loss={loss:.3f} "
            f"Acc={acc:.3f} "
            f"P={p:.3f} "
            f"R={r:.3f} "
            f"F1={f1:.3f}"
        )

    # Interactive prediction (like Naive Bayes)
    print("\nEnter sentences to test (empty line to stop):")
    while True:
        s = input("> ")
        if not s:
            break

        s = s.replace(word, f"<target> {word} </target>")
        result = predict_sentences(
            model, [s], word2idx, idx2sense, 20, device
        )
        print("Predicted Sense:", result[0][1])
