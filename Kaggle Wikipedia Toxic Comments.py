import io
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 75000
max_length = 120
embedding_dim = 16
trunc_type = "post"
oov_tok = "<OOV>"
##num_epochs = 8

def run_all(num_epochs):
    
    toxicity_vector, comments, padded, word_index = params("train_path")
    toxicity_vector = to_binary_choice(toxicity_vector)
    
    print(toxicity_vector.shape)
    print(comments.shape)
    print(padded.shape)
    
    return


    model, history = compile_model(padded, toxicity_vector, num_epochs)
    export_semantics(model, word_index)
    
    return model, history
    
def to_binary_choice(vec):
    
    
    binary_vector = []
    
    for vector in vec:
        good_comment = True
        for i in vector:
            if i == 1:
                good_comment = False
                
        if good_comment == True:
            binary_vector.append(0)
        if good_comment == False:
            binary_vector.append(1)
            
    return np.asarray(binary_vector)

def params(fp):
    
    if fp == "train_path":
        file_path = "C:/Users/jorda/Documents/Year 3 Work/Programming/ML/tensorflow/Wikipedia Toxic Comments/train.csv"
    else:
        file_path = fp
    toxicity_vector, comments = read_file(file_path)
    padded, word_index = process_comments(comments)
    
    return toxicity_vector, comments, padded, word_index

def import_test(fp1, fp2):
    
    lines1 = []
    reader = csv.reader(open(fp1, 'r', encoding='latin-1'))
    for line in reader:
        lines1.append(line)     
    lines1.pop(0)
    
    
def read_file(fp):
    
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
    
    #np.reshape(toxicity_vector, ((6, len(toxicity_vector))))
    #np.reshape(comments, ((1, len(comments))))
    
    toxicity_vector, comments = np.asarray(toxicity_vector), np.asarray(comments)
    
    return toxicity_vector, comments

def process_comments(comments):
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(comments)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(comments)
    padded = pad_sequences(sequences, maxlen = max_length)
    
    return padded, word_index

def compile_model(padded, toxicity_vector, num_epochs):
    
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.99):
                print("\nReached 99% accuracy so stopped training.")
                self.model.stop_training = True
                
    callbacks = myCallback()
    
    model = tf.keras.Sequential([
        
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        #tf.keras.layers.Dense(36, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') # FOR BINARY CHOICE
        #tf.keras.layers.Dense(6, activation='softmax') # FOR NON-BINARY CHOICE
        
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # FOR BINARY CHOICE
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # FOR NON_BINARY CHOICE
    model.summary()
    
    history = model.fit(padded, toxicity_vector, epochs = num_epochs, callbacks=[callbacks])
    
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

"""
Importing of Test Data
"""
    
    