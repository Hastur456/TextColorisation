from google.colab import files

!pip install -q kaggle

files.upload() # установка токена

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d "simaanjali/emotion-analysis-based-on-text"

!unzip emotion-analysis-based-on-text.zip

!rm emotion-analysis-based-on-text.zip

!pip install -q torchmetrics

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from google.colab import drive
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import numpy as np
from tqdm.autonotebook import tqdm


nltk.download("punkt_tab")

df = pd.read_csv("/content/emotion_sentimen_dataset.csv", sep=",", header=None)
df.columns = ["idx", "text", "emotions"]
df.drop(df.index[0], inplace=True)
del df["idx"]

labels_count = df["emotions"].value_counts()

#data visualisation
plt.bar(list(labels_count.keys()), list(labels_count.values))
plt.show()

emotions_dict = {}
for idx, em in enumerate(list(labels_count.keys())):
    emotions_dict[em] = idx

df["emotions_nums"] = df["emotions"].replace(emotions_dict)

X = df["text"].values
y = df["emotions_nums"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, train_size=0.05)

vectorizer = TfidfVectorizer(max_features=15000)
vectorizer.fit(X)

X_train = vectorizer.transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

class ColorisationTextClassifier(nn.Module):
  def __init__(self,
               vec_len: int,
               num_classes: int,
               hidden_size_LSTM = 128
               ):
    super().__init__()
    self.LSTM_1 = nn.LSTM(input_size=vec_len, hidden_size=hidden_size_LSTM, batch_first=True, bidirectional=True, dropout=0.2)
    self.outputs = nn.Linear(hidden_size_LSTM, num_classes)

  def forward(self, x):
    x = x.unsqueeze(1)
    outputs, (hn, cn) = self.LSTM_1(x)
    x = self.outputs(hn[-1])
    return x

vec_len = len(vectorizer.vocabulary_)
num_classes = len(emotions_dict)

model = ColorisationTextClassifier(vec_len, num_classes)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=42e-4)

EPOCHS = 100
total_loss = []
train_loss = []

for epoch in tqdm(range(EPOCHS)):
  for X_batch, y_batch in train_dataloader:
    model.train()
    optimizer.zero_grad()

    logits = model(X_batch.to(DEVICE))
    loss = criterion(logits, y_batch.to(DEVICE))
    train_loss.append(loss.detach().cpu().numpy())
    loss.backward()

    optimizer.step()
  print("Epoch: {}, Loss: {}".format(epoch, np.mean(train_loss)))
  total_loss.append(np.mean(train_loss))
  train_loss.clear()

plt.figure(figsize=(15, 6))
plt.plot(total_loss)
plt.title("Text Colorisation Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

predictions = []
targets = []

model.eval()
with torch.no_grad():
  for x, target in tqdm(test_dataloader):
    outputs = torch.argmax(model(x.to(DEVICE)), 1)
    predictions.append(outputs.detach())
    targets.append(target.detach())

predictions = torch.cat(predictions).cpu().numpy()
targets = torch.cat(targets).cpu().numpy()

print(f"Accuracy: {accuracy_score(targets, predictions)}")

print(classification_report(targets, predictions, target_names=list(emotions_dict.keys())))
