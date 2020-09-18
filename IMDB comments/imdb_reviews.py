import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import io
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"
    
def import_data():
    
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train_data, test_data = imdb['train'], imdb['test']
    
    training_sentences = []
    training_labels = []
    
    testing_sentences = []
    testing_labels = []
    
    for s, l in train_data:
        training_sentences.append(str(s.numpy()))
        training_labels.append(l.numpy())
        
    for s, l in test_data:
        testing_sentences.append(str(s.numpy()))
        testing_labels.append(l.numpy())
        
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
    
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length)
    
    
    print(training_labels_final.shape)
    print(padded.shape)
    
    return padded, training_labels_final, testing_padded, testing_labels_final, word_index

def train_imdb(padded, training_labels_final, testing_padded, testing_labels_final, num_epochs):
    
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy')>0.99):
                print("\nReached 99% accuracy so stopping training.")
                self.model.stop_training = True
                
    callbacks = myCallback()
    
    model = tf.keras.Sequential([
        
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    history = model.fit(padded, training_labels_final, validation_data=(testing_padded, testing_labels_final), epochs=num_epochs, callbacks=[callbacks])
    
    return model, history
    
def export_semantics(model, word_index):
    
    e = model.layers[0]
    weights = e.get_weights()[0]
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    out_v = io.open('/Users/jorda/Documents/Year 2 Work/tensorflow/IMDB - Good or Bad/results/vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('/Users/jorda/Documents/Year 2 Work/tensorflow/IMDB - Good or Bad/results/meta.tsv', 'w', encoding='utf-8')
    
    for word_num in range(1, vocab_size):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
        
    out_v.close()  
    out_m.close()
    
    return

p, tr, tep, te, word_index = import_data()
model, history = train_imdb(p, tr, tep, te, 50)
export_semantics(model, word_index)