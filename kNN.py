import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# =============================
# k-Nearest Neighbors Model
# =============================
class KNNWSD:
    def __init__(self, k=5):
        self.k = k
        # We initialize the vectorizer here so it persists between train and predict
        self.vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r"\b\w+\b"
        )
        self.classifier = KNeighborsClassifier(
            n_neighbors=k,
            metric="cosine",
            weights="distance"
        )

    def train(self, sentences, senses):
        # 1. Vectorize the training sentences
        X_train = self.vectorizer.fit_transform(sentences)

        # 2. Train the kNN classifier
        self.classifier.fit(X_train, senses)

    def predict(self, sentence):
        # 1. Transform the single sentence using the already fitted vectorizer
        # Note: We pass [sentence] because transform expects a list/iterable
        X_test = self.vectorizer.transform([sentence])

        # 2. Predict using the classifier
        # Returns a list, so we take the first element [0]
        return self.classifier.predict(X_test)[0]


# =============================
# Evaluation
# =============================
def evaluate(model, sentences, senses):
    y_true = []
    y_pred = []

    # Predict one by one to match the 'predict' method signature
    for s, gold in zip(sentences, senses):
        y_true.append(gold)
        y_pred.append(model.predict(s))

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
    try:
        with open(f"Sentence/{word}sentence.inp", "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f]

        with open(f"Sense/{word}sense.inp", "r", encoding="utf-8") as f:
            senses = [line.strip() for line in f]
    except FileNotFoundError:
        print("Error: Files not found.")
        exit()

    assert len(sentences) == len(senses)

    # Target-word marking
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

    # Train kNN
    # You can change k here (e.g., k=3, k=5, k=10)
    knn = KNNWSD(k=5)
    knn.train(train_sent, train_sense)

    # Evaluate
    acc, p, r, f1 = evaluate(knn, test_sent, test_sense)

    print("\nKNN Evaluation:")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall   : {r:.3f}")
    print(f"F1-score : {f1:.3f}")

    # Interactive Prediction
    print("\nEnter sentences to test (empty line to stop):")
    while True:
        s = input("> ")
        if not s:
            break

        s = s.replace(word, f"<target> {word} </target>")
        print("Predicted Sense:", knn.predict(s))