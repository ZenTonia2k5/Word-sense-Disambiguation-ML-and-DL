import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =============================
# Evaluation
# =============================
def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
    return acc, precision, recall, f1

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    word = input("Enter target word: ").strip().lower()

    # Load data
    with open(f"Sentence/{word}sentence.inp", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f]

    with open(f"Sense/{word}sense.inp", "r", encoding="utf-8") as f:
        senses = [line.strip() for line in f]

    assert len(sentences) == len(senses)

    # Target-word marking (same as NB & LSTM)
    sentences = [
        s.replace(word, f"<target> {word} </target>")
        for s in sentences
    ]

    # Train / test split (80 / 20)
    data = list(zip(sentences, senses))
    random.shuffle(data)
    split = int(0.8 * len(data))

    train_data = data[:split]
    test_data = data[split:]

    train_sent, train_sense = zip(*train_data)
    test_sent, test_sense = zip(*test_data)

    # =============================
    # Vectorization (Bag-of-Words)
    # =============================
    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"\b\w+\b"
    )

    X_train = vectorizer.fit_transform(train_sent)
    X_test = vectorizer.transform(test_sent)

    # =============================
    # KNN Classifier
    # =============================
    k = 5
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine",
        weights="distance"
    )

    knn.fit(X_train, train_sense)

    # =============================
    # Evaluation
    # =============================
    y_pred = knn.predict(X_test)
    acc, p, r, f1 = evaluate(test_sense, y_pred)

    print("\nKNN Evaluation:")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall   : {r:.3f}")
    print(f"F1-score : {f1:.3f}")

    # =============================
    # Interactive Prediction
    # =============================
    print("\nEnter sentences to test (empty line to stop):")
    while True:
        s = input("> ")
        if not s:
            break

        s = s.replace(word, f"<target> {word} </target>")
        x = vectorizer.transform([s])
        pred = knn.predict(x)[0]
        print("Predicted Sense:", pred)
