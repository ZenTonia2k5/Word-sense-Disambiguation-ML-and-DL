import math
import random
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =============================
# Naive Bayes Model
# =============================
class NaiveBayesWSD:
    def __init__(self):
        self.class_priors = {}
        self.word_probs = {}
        self.vocab = set()
        self.classes = set()

    def tokenize(self, sentence):
        return sentence.lower().split()

    def train(self, sentences, senses):
        doc_count = len(sentences)
        sense_counts = Counter(senses)
        self.classes = set(senses)

        # Log priors P(sense)
        self.class_priors = {
            s: math.log(c / doc_count)
            for s, c in sense_counts.items()
        }

        # Word counts per sense
        word_counts = defaultdict(Counter)
        total_words = defaultdict(int)

        for sentence, sense in zip(sentences, senses):
            tokens = self.tokenize(sentence)
            for tok in tokens:
                word_counts[sense][tok] += 1
                total_words[sense] += 1
                self.vocab.add(tok)

        vocab_size = len(self.vocab)

        # Likelihoods with Laplace smoothing
        self.word_probs = {}
        for sense in self.classes:
            self.word_probs[sense] = {}
            for word in self.vocab:
                count = word_counts[sense][word]
                self.word_probs[sense][word] = math.log(
                    (count + 1) / (total_words[sense] + vocab_size)
                )

    def predict(self, sentence):
        tokens = self.tokenize(sentence)
        scores = {}

        for sense in self.classes:
            score = self.class_priors[sense]
            for tok in tokens:
                if tok in self.vocab:
                    score += self.word_probs[sense][tok]
            scores[sense] = score

        return max(scores, key=scores.get)

# =============================
# Evaluation
# =============================
def evaluate(model, sentences, senses):
    y_true = []
    y_pred = []

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

    with open(f"Sentence/{word}sentence.inp", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f]

    with open(f"Sense/{word}sense.inp", "r", encoding="utf-8") as f:
        senses = [line.strip() for line in f]

    assert len(sentences) == len(senses)

    # Target-word marking (same as LSTM)
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

    # Train NB
    nb = NaiveBayesWSD()
    nb.train(train_sent, train_sense)

    # Evaluate
    acc, p, r, f1 = evaluate(nb, test_sent, test_sense)

    print("\nNaive Bayes Evaluation:")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall   : {r:.3f}")
    print(f"F1-score : {f1:.3f}")

    # Interactive prediction (multiple sentences)
    print("\nEnter sentences to test (empty line to stop):")
    while True:
        s = input("> ")
        if not s:
            break

        s = s.replace(word, f"<target> {word} </target>")
        print("Predicted Sense:", nb.predict(s))
