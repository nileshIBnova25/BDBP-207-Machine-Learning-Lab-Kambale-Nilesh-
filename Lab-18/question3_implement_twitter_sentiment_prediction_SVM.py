import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load the data
# Use encoding='latin1' or 'utf-8' depending on the file save format
df = pd.read_csv('Tweets.csv')

# 2. Map the correct columns
# Based on your error, 'airline_sentiment' is the label and 'text' is the feature
X_raw = df['text']
y = df['airline_sentiment']

# 3. Preprocess / Vectorize
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X = tfidf.fit_transform(X_raw)

# 4. Split data (90% train, 10% test as per Lab 18 instructions)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# 5. Compare Kernels

def main():
    for k in ['linear', 'rbf', 'poly']:
        model = SVC(kernel=k)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Kernel: {k:8} | Accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()