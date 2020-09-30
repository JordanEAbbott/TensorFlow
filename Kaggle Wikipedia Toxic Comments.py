import io
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

vocab_size = 75000
max_length = 120
embedding_dim = 64
trunc_type = "post"
oov_tok = "<OOV>"

def run_all(num_lines):
    
    num_epochs = 16
    
    toxicity_vector, comments, padded, word_index, tokenizer = params("train_path", True, num_lines)

    model, history = compile_model(padded, toxicity_vector, num_epochs)
    export_semantics(model, word_index)
    
    return model, history

def params(fp, train, num_lines):
    
    if fp == "train_path":
        file_path = "C:/Users/jorda/Documents/Year 3 Work/Programming/ML/tensorflow/Wikipedia Toxic Comments/train.csv"
    else:
        file_path = fp
    toxicity_vector, comments = read_file(file_path, num_lines)
    
    if train:
        padded, word_index, tokenizer = process_comments(comments)
        return toxicity_vector, comments, padded, word_index, tokenizer
    
    return toxicity_vector, comments
    
def read_file(fp, n):
    
    lines = []
    reader = csv.reader(open(fp, 'r', encoding='latin-1'))
    for line in reader:
        lines.append(line)
     
    lines.pop(0)
    
    toxicity_vector = []
    comments = []
    
    for line in lines:
        tmp = [int(i) for i in line[-6:]]
        toxicity_vector.append(tmp)
        comments.append(line[1])
    
    toxicity_vector, comments = np.asarray(toxicity_vector), np.asarray(comments)
    
    return toxicity_vector[:n], comments[:n]

def process_comments(comments):
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(comments)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(comments)
    padded = pad_sequences(sequences, maxlen = max_length)
    
    return padded, word_index, tokenizer

def compile_model(padded, toxicity_vector, num_epochs):
    
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.95):
                print("\nReached 95% accuracy so stopped training.")
                self.model.stop_training = True
                
    callbacks = myCallback()
    
    model = tf.keras.Sequential([
        
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(LSTM(100, return_sequences=True, dropout=0.50), merge_mode='concat'),
        tf.keras.layers.TimeDistributed(Dense(100,activation='relu')),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(6, activation='sigmoid')
        
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # FOR NON_BINARY CHOICE
    model.summary()
    
    history = model.fit(padded, toxicity_vector, epochs = num_epochs, batch_size = 256, callbacks=[callbacks])
    
    return model, history

def model_history(history):
    
    acc = history.history['accuracy']
    #val_acc = history.history['val_accuracy']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    #plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    
    plt.show()
    
def export_semantics(model, word_index):
    
    e = model.layers[0]
    weights = e.get_weights()[0]
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    out_v = io.open("/Users/jorda/Documents/Year 3 Work/Programming/ML/tensorflow/Wikipedia Toxic Comments/vecs.tsv", 'w', encoding='utf-8')
    out_m = io.open("/Users/jorda/Documents/Year 3 Work/Programming/ML/tensorflow/Wikipedia Toxic Comments/meta.tsv", 'w', encoding='utf-8')
    
    for word_num in range(1, vocab_size):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
        
    out_v.close()
    out_m.close()
    
    return
