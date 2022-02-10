from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from statistics import mean
import numpy as np
import pandas as pd
import json
import re
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict, cross_val_score
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout
from sklearn.model_selection import train_test_split
import pydotplus
from tensorflow.keras.utils import plot_model
from keras.callbacks import EarlyStopping

seed = 42
np.random.seed(seed)


# Opening JSON file
f = open('sarcasm_data.json')
 
# returns JSON object as
# a dictionary
data = json.load(f)

# Iterating through the json
# list

work_set = {}

i = 0
for key, value in data.items():
    work_set[i] = {k:value[k] for k in ("context", "sarcasm")}
    i += 1


# Closing file
f.close()

df = pd.DataFrame(work_set).T  
df['context'] = df['context'].str.join(" ")
le = LabelEncoder()
num_classes=2
df['sarcasm'] = le.fit_transform(df['sarcasm'])

#Preprocessing
stop = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
df['context'] = df['context'].str.lower()
df['context'] = df['context'].apply(lambda x : " ".join(re.findall('[\w]+',x)))
df['context'] = df['context'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['context'] = df['context'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()])) 

#Baseline model
#Vectorization
#N-grams parameter
ngrams = 1

#Vectorize words into n-grams
ngram_counter = CountVectorizer(ngram_range=(ngrams, ngrams), analyzer='word')

X = ngram_counter.fit_transform(df['context']).toarray()
y = df['sarcasm'].astype('int')

#SVC
#Classifier
classifier = SVC(kernel='rbf', C=1, gamma=0.1)

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, y)
acc = []
rec = []
pre = []
f1 = []

for train_index, test_index in skf.split(X, y):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     model = classifier.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     acc.append(accuracy_score(y_test, y_pred))
     rec.append(recall_score(y_test, y_pred))
     pre.append(precision_score(y_test, y_pred))
     f1.append(f1_score(y_test, y_pred))

acc_mean = mean(acc)
rec_mean = mean(rec)
pre_mean = mean(pre)
f1_mean = mean(f1)

print('Precision: %.3f' % pre_mean)
print('Recall: %.3f' % rec_mean)
print('Accuracy: %.3f' % acc_mean)
print('F1 Score: %.3f' % f1_mean)

#DL
#tokenization
t = Tokenizer(oov_token='<UNK>')
t.fit_on_texts(df['context'])
t.word_index['<PAD>'] = 0

max([(k, v) for k, v in t.word_index.items()], key = lambda x:x[1]), min([(k, v) for k, v in t.word_index.items()], key = lambda x:x[1]), t.word_index['<UNK>']
train_sequences = t.texts_to_sequences(df['context'])

MAX_SEQUENCE_LENGTH = 100
# pad dataset to a maximum review length in words
X = sequence.pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)

VOCAB_SIZE = len(t.word_index)
EMBED_SIZE = 64
EPOCHS=20 
BATCH_SIZE=32
MAX_SEQUENCE_LENGTH = 100

#MLP
acc = []
rec = []
pre = []
f1 = []

for train_index, test_index in skf.split(X, y):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     model = tf.keras.Sequential([
     tf.keras.layers.Dense(64, activation='sigmoid', input_dim = 100, kernel_regularizer = regularizers.l2(1e-5)),
     tf.keras.layers.Dense(32, activation='sigmoid'),
     tf.keras.layers.Dense(1, activation='sigmoid')])
     #model.summary()  
     #plot_model(model, to_file='MLP_plot.png', show_shapes=True, show_layer_names=False)
     #es = EarlyStopping(monitor='loss', mode='min', patience = 3)
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     model.fit(X_train, y_train,epochs=EPOCHS)
     y_pred = (model.predict(X_test) > 0.5).astype("int32")
     acc.append(accuracy_score(y_test, y_pred))
     rec.append(recall_score(y_test, y_pred))
     pre.append(precision_score(y_test, y_pred))
     f1.append(f1_score(y_test, y_pred))

acc_mean = mean(acc)
rec_mean = mean(rec)
pre_mean = mean(pre)
f1_mean = mean(f1)

print('Precision: %.3f' % pre_mean)
print('Recall: %.3f' % rec_mean)
print('Accuracy: %.3f' % acc_mean)
print('F1 Score: %.3f' % f1_mean)

#LSTM
acc = []
rec = []
pre = []
f1 = []

for train_index, test_index in skf.split(X, y):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     model = tf.keras.Sequential([
     tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAX_SEQUENCE_LENGTH),
     tf.keras.layers.LSTM(64, return_sequences=False),
     tf.keras.layers.Dense(32, activation='sigmoid'),
     tf.keras.layers.Dense(1, activation='sigmoid')])
     #model.summary()     
     #plot_model(model, to_file='LSTM_plot.png', show_shapes=True, show_layer_names=False)
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     model.fit(X_train, y_train, epochs=EPOCHS)
     y_pred = (model.predict(X_test) > 0.5).astype("int32")
     acc.append(accuracy_score(y_test, y_pred))
     rec.append(recall_score(y_test, y_pred))
     pre.append(precision_score(y_test, y_pred))
     f1.append(f1_score(y_test, y_pred))

acc_mean = mean(acc)
rec_mean = mean(rec)
pre_mean = mean(pre)
f1_mean = mean(f1)

print('Precision: %.3f' % pre_mean)
print('Recall: %.3f' % rec_mean)
print('Accuracy: %.3f' % acc_mean)
print('F1 Score: %.3f' % f1_mean)

#CNN
acc = []
rec = []
pre = []
f1 = []

for train_index, test_index in skf.split(X, y):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     model = tf.keras.Sequential([
     tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAX_SEQUENCE_LENGTH),
     tf.keras.layers.Conv1D(filters = 32, kernel_size = 32, padding = 'same',activation='relu'),
     tf.keras.layers.GlobalMaxPooling1D(),
     tf.keras.layers.Dense(16, activation='sigmoid'),
     tf.keras.layers.Dense(1, activation='sigmoid')])
     # model.summary()     
     # plot_model(model, to_file='CNN_plot.png', show_shapes=True, show_layer_names=False)
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     model.fit(X_train, y_train, epochs=EPOCHS)
     y_pred = (model.predict(X_test) > 0.5).astype("int32")
     acc.append(accuracy_score(y_test, y_pred))
     rec.append(recall_score(y_test, y_pred))
     pre.append(precision_score(y_test, y_pred))
     f1.append(f1_score(y_test, y_pred))

acc_mean = mean(acc)
rec_mean = mean(rec)
pre_mean = mean(pre)
f1_mean = mean(f1)

print('Precision: %.3f' % pre_mean)
print('Recall: %.3f' % rec_mean)
print('Accuracy: %.3f' % acc_mean)
print('F1 Score: %.3f' % f1_mean)

#BiLSTM
acc = []
rec = []
pre = []
f1 = []
dropout_rate= 0.4

for train_index, test_index in skf.split(X, y):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     model = tf.keras.Sequential([
     tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAX_SEQUENCE_LENGTH),
     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(EMBED_SIZE, return_sequences=False), merge_mode='concat'),
     tf.keras.layers.Dense(EMBED_SIZE, activation='relu'),
     tf.keras.layers.Dense(1, activation='sigmoid')])
     # model.summary()   
     # plot_model(model, to_file='BiLSTM_plot.png', show_shapes=True, show_layer_names=False)
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     model.fit(X_train, y_train, epochs=EPOCHS)
     y_pred = (model.predict(X_test) > 0.5).astype("int32")
     acc.append(accuracy_score(y_test, y_pred))
     rec.append(recall_score(y_test, y_pred))
     pre.append(precision_score(y_test, y_pred))
     f1.append(f1_score(y_test, y_pred))

acc_mean = mean(acc)
rec_mean = mean(rec)
pre_mean = mean(pre)
f1_mean = mean(f1)

print('Precision: %.3f' % pre_mean)
print('Recall: %.3f' % rec_mean)
print('Accuracy: %.3f' % acc_mean)
print('F1 Score: %.3f' % f1_mean)

#CNN-Large
acc = []
rec = []
pre = []
f1 = []
dropout_rate= 0.4

for train_index, test_index in skf.split(X, y):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     model = tf.keras.Sequential([
     tf.keras.layers.Embedding(VOCAB_SIZE, 128, input_length=MAX_SEQUENCE_LENGTH),
     tf.keras.layers.Conv1D(filters = 128, kernel_size = 8, padding = 'same', activation='relu'),
     tf.keras.layers.MaxPooling1D(pool_size = 2),
     tf.keras.layers.Dropout(rate=dropout_rate),
     tf.keras.layers.Conv1D(filters = 64, kernel_size = 8, padding = 'same', activation='relu'),
     tf.keras.layers.GlobalMaxPooling1D(),   
     tf.keras.layers.Dropout(rate=dropout_rate),
     tf.keras.layers.Conv1D(filters = 32, kernel_size = 8, padding = 'same', activation='relu'),
     tf.keras.layers.MaxPooling1D(pool_size = 2),
     tf.keras.layers.Dropout(rate=dropout_rate),
     tf.keras.layers.Conv1D(filters = 16, kernel_size = 8, padding = 'same', activation='relu'),
     tf.keras.layers.GlobalMaxPooling1D(),
     tf.keras.layers.Dropout(rate=dropout_rate),
     tf.keras.layers.Dense(32, activation='relu'),
     tf.keras.layers.Dropout(rate=dropout_rate),
     tf.keras.layers.Dense(1, activation='sigmoid')])
     model.summary()   
     plot_model(model, to_file='CNN-Large_plot.png', show_shapes=True, show_layer_names=False)
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     model.fit(X_train, y_train, epochs=EPOCHS)
     y_pred = (model.predict(X_test) > 0.5).astype("int32")
     acc.append(accuracy_score(y_test, y_pred))
     rec.append(recall_score(y_test, y_pred))
     pre.append(precision_score(y_test, y_pred))
     f1.append(f1_score(y_test, y_pred))

acc_mean = mean(acc)
rec_mean = mean(rec)
pre_mean = mean(pre)
f1_mean = mean(f1)

print('Precision: %.3f' % pre_mean)
print('Recall: %.3f' % rec_mean)
print('Accuracy: %.3f' % acc_mean)
print('F1 Score: %.3f' % f1_mean)

#CNN-LSTM
acc = []
rec = []
pre = []
f1 = []
dropout_rate= 0.4

for train_index, test_index in skf.split(X, y):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     model = tf.keras.Sequential([
     tf.keras.layers.Embedding(VOCAB_SIZE, 128, input_length=MAX_SEQUENCE_LENGTH),
     tf.keras.layers.Conv1D(filters=64, kernel_size=8, padding='same', activation='relu'),
     tf.keras.layers.MaxPooling1D(pool_size=2),
    #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16), merge_mode='concat'),
    #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(16, activation='relu'),
    # tf.keras.layers.Dropout(0.8),
     tf.keras.layers.Dense(1, activation='sigmoid')])
     model.summary()
     plot_model(model, to_file='CNN-LSTM_plot.png', show_shapes=True, show_layer_names=False)
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     model.fit(X_train, y_train, epochs=EPOCHS)
     y_pred = (model.predict(X_test) > 0.5).astype("int32")
     acc.append(accuracy_score(y_test, y_pred))
     rec.append(recall_score(y_test, y_pred))
     pre.append(precision_score(y_test, y_pred))
     f1.append(f1_score(y_test, y_pred))

acc_mean = mean(acc)
rec_mean = mean(rec)
pre_mean = mean(pre)
f1_mean = mean(f1)

print('Precision: %.3f' % pre_mean)
print('Recall: %.3f' % rec_mean)
print('Accuracy: %.3f' % acc_mean)
print('F1 Score: %.3f' % f1_mean)

#LSTM-Attention
from tensorflow.keras.layers import Layer
import keras.backend as K

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()
    
acc = []
rec = []
pre = []
f1 = []
dropout_rate= 0.4

for train_index, test_index in skf.split(X, y):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     model = Sequential()
     model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAX_SEQUENCE_LENGTH))
     model.add(LSTM(EMBED_SIZE, activation="relu", return_sequences=True))
     model.add(attention()) 
     model.add(Dense(32, activation='relu'))
     model.add(Dense(1, activation='sigmoid'))
     model.summary()
     plot_model(model, to_file='Attn-LSTM_plot.png', show_shapes=True, show_layer_names=False)
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     model.fit(X_train, y_train, epochs=EPOCHS)
     y_pred = (model.predict(X_test) > 0.5).astype("int32")
     acc.append(accuracy_score(y_test, y_pred))
     rec.append(recall_score(y_test, y_pred))
     pre.append(precision_score(y_test, y_pred))
     f1.append(f1_score(y_test, y_pred))

acc_mean = mean(acc)
rec_mean = mean(rec)
pre_mean = mean(pre)
f1_mean = mean(f1)

print('Precision: %.3f' % pre_mean)
print('Recall: %.3f' % rec_mean)
print('Accuracy: %.3f' % acc_mean)
print('F1 Score: %.3f' % f1_mean)

#GloVe
path_to_glove_file = 'glove.6B.100d.txt'

embeddings_index = {}
with open(path_to_glove_file, encoding="utf8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

vectorizer = TextVectorization(max_tokens=3000, output_sequence_length=100)
text_ds = tf.data.Dataset.from_tensor_slices(df['context']).batch(128)
vectorizer.adapt(text_ds)

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False)

#CNN
acc = []
rec = []
pre = []
f1 = []

for train_index, test_index in skf.split(X, y):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     model = Sequential()
     model.add(embedding_layer)
     model.add(Conv1D(filters=128, kernel_size=64, padding='same', activation='relu'))
     model.add(MaxPooling1D(pool_size=2))
     model.add(Dropout(0.7))
     model.add(Conv1D(filters=64, kernel_size=64, padding='same', activation='relu'))
     model.add(MaxPooling1D(pool_size=2))
     model.add(Dropout(0.7))
     model.add(Conv1D(filters=32, kernel_size=64, padding='same', activation='relu'))
     model.add(GlobalMaxPooling1D())
     model.add(Dropout(0.7))
     model.add(Dense(256, activation='relu'))
     model.add(Dropout(0.7))
     model.add(Dense(1, activation='sigmoid'))
     model.summary()     
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     model.fit(X_train, y_train, epochs=EPOCHS)
     y_pred = (model.predict(X_test) > 0.5).astype("int32")
     acc.append(accuracy_score(y_test, y_pred))
     rec.append(recall_score(y_test, y_pred))
     pre.append(precision_score(y_test, y_pred))
     f1.append(f1_score(y_test, y_pred))

acc_mean = mean(acc)
rec_mean = mean(rec)
pre_mean = mean(pre)
f1_mean = mean(f1)

print('Precision: %.3f' % pre_mean)
print('Recall: %.3f' % rec_mean)
print('Accuracy: %.3f' % acc_mean)
print('F1 Score: %.3f' % f1_mean)

#BiLSTM
acc = []
rec = []
pre = []
f1 = []

for train_index, test_index in skf.split(X, y):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     model = tf.keras.Sequential([
     embedding_layer,
     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(EMBED_SIZE, return_sequences=False), merge_mode='concat'),
     tf.keras.layers.Dense(EMBED_SIZE, activation='relu'),
     tf.keras.layers.Dropout(rate=0.4),
     tf.keras.layers.Dense(1, activation='sigmoid')])
 
     model.summary()    
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     model.fit(X_train, y_train, epochs=EPOCHS)
     y_pred = (model.predict(X_test) > 0.5).astype("int32")
     acc.append(accuracy_score(y_test, y_pred))
     rec.append(recall_score(y_test, y_pred))
     pre.append(precision_score(y_test, y_pred))
     f1.append(f1_score(y_test, y_pred))

acc_mean = mean(acc)
rec_mean = mean(rec)
pre_mean = mean(pre)
f1_mean = mean(f1)

print('Precision: %.3f' % pre_mean)
print('Recall: %.3f' % rec_mean)
print('Accuracy: %.3f' % acc_mean)
print('F1 Score: %.3f' % f1_mean)

#CNN-LSTM
acc = []
rec = []
pre = []
f1 = []

for train_index, test_index in skf.split(X, y):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     model = tf.keras.Sequential([
     embedding_layer,
     tf.keras.layers.Conv1D(filters=128, kernel_size=64, padding='same', activation='relu'),
     tf.keras.layers.MaxPooling1D(pool_size=2),
    #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(EMBED_SIZE), merge_mode='concat'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(EMBED_SIZE, activation='relu'),
    # tf.keras.layers.Dropout(0.8),
     tf.keras.layers.Dense(1, activation='sigmoid')])
     model.summary()
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     model.fit(X_train, y_train, epochs=EPOCHS)
     y_pred = (model.predict(X_test) > 0.5).astype("int32")
     acc.append(accuracy_score(y_test, y_pred))
     rec.append(recall_score(y_test, y_pred))
     pre.append(precision_score(y_test, y_pred))
     f1.append(f1_score(y_test, y_pred))

acc_mean = mean(acc)
rec_mean = mean(rec)
pre_mean = mean(pre)
f1_mean = mean(f1)

print('Precision: %.3f' % pre_mean)
print('Recall: %.3f' % rec_mean)
print('Accuracy: %.3f' % acc_mean)
print('F1 Score: %.3f' % f1_mean)

#Transformers
#BERT
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback

# Define pretrained tokenizer and model
model_name = "bert-base-uncased"

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

acc = []
rec = []
pre = []
f1 = []

for train_index, test_index in skf.split(X, y):
     train_df = df.iloc[train_index]
     val_df = df.iloc[test_index]
     tokenizer = BertTokenizer.from_pretrained(model_name)
    # Preprocess data
     Xt = list(train_df["context"])
     yt = list(train_df["sarcasm"])
    
     X_train, X_val, y_train, y_val = train_test_split(Xt, yt, test_size=0.2)
     X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=100)
     X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=100)
     X_test_tokenized = tokenizer(list(val_df["context"]), padding=True, truncation=True, max_length=100)

     train_dataset = Dataset(X_train_tokenized, y_train)
     val_dataset = Dataset(X_val_tokenized, y_val)
     test_dataset = Dataset(X_test_tokenized, list(val_df["sarcasm"]))
    
     model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
     
    # Define Trainer
     args = TrainingArguments(
     output_dir="output",
     evaluation_strategy="steps",
     eval_steps=500,
     per_device_train_batch_size=8,
     per_device_eval_batch_size=8,
     num_train_epochs=EPOCHS,
     seed=0,
     load_best_model_at_end=True,)
     trainer = Trainer(
     model=model,
     args=args,
     train_dataset=train_dataset,
     eval_dataset=val_dataset,
     compute_metrics=compute_metrics,
     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],)
    
     trainer.train()
    
     raw_pred, _, _ = trainer.predict(test_dataset)

     y_pred = np.argmax(raw_pred, axis=1)
    
     acc.append(accuracy_score(y_test, y_pred))
     rec.append(recall_score(y_test, y_pred))
     pre.append(precision_score(y_test, y_pred))
     f1.append(f1_score(y_test, y_pred))

acc_mean = mean(acc)
rec_mean = mean(rec)
pre_mean = mean(pre)
f1_mean = mean(f1)

print('Precision: %.3f' % pre_mean)
print('Recall: %.3f' % rec_mean)
print('Accuracy: %.3f' % acc_mean)
print('F1 Score: %.3f' % f1_mean)
