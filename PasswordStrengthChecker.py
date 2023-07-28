import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import getpass

data = pd.read_csv(r'C:\Users\denicola\Downloads\training.csv')

#The dataset has two columns; password and strength. In the strength column:
#0 means: the password’s strength is weak;
#1 means: the password’s strength is medium;
#2 means: the password’s strength is strong;

data = data.dropna()
data["strength"] = data["strength"].map({0: "Weak", 1: "Medium", 2: "Strong"})


x = np.array(data["password"])
y = np.array(data["strength"])

vectorizer = TfidfVectorizer(analyzer='char')
x = vectorizer.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.05, random_state=42)

model = RandomForestClassifier()

batch_size = 100
num_batches = xtrain.shape[0] // batch_size

with tqdm(total=num_batches, desc="Training model", ncols=100) as pbar:
    total_data_used = 0
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size

        x_batch = xtrain[start_idx:end_idx]
        y_batch = ytrain[start_idx:end_idx]

        model.fit(x_batch, y_batch)
        total_data_used += x_batch.shape[0]

        pbar.update()

        pbar.set_description(f"Training model: {total_data_used}/{xtrain.shape[0]}")

        
# Valutazione delle prestazioni del modello sul test set
ypred = model.predict(xtest)

# Calcolo di diverse metriche
accuracy = accuracy_score(ytest, ypred)
precision = precision_score(ytest, ypred, average='weighted')
recall = recall_score(ytest, ypred, average='weighted')
f1 = f1_score(ytest, ypred, average='weighted')

print("Model score:", model.score(xtest, ytest))
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

user = getpass.getpass("Enter Password: ")
data = vectorizer.transform([user]).toarray()
output = model.predict(data)
print(output)

