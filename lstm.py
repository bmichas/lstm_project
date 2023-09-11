from tensorflow.keras.layers import Dense, LSTM, Input, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
from random import randint
import numpy as np

from function import unpickle


    

file_path = 'data/tokenizer_080423.pkl'
tokenizer = unpickle(file_path)

class PredictionCallback(tf.keras.callbacks.Callback):    
    def _preprocess_input_text(self, text, max_len):
        text = text.split()
        sequence = text[-max_len:]
        vector_sequences = []
        vector_sequence = []
        for word in sequence:
            vector_sequence.append(word)

        vector_sequences.append(vector_sequence)
        vector_sequences = tokenizer.texts_to_sequences(vector_sequences)
        
        vector_diff = max_len - len(vector_sequences[0])
        if not vector_diff:
            return np.array(vector_sequences, dtype=np.int32)
        
        for _ in range(vector_diff):
            vector_sequences[0].insert(0, randint(0, len(tokenizer.word_index)))

        return np.array(vector_sequences, dtype=np.int32)
            

    def on_epoch_end(self, epoch, logs={}):
        """SAVING MODEL"""
        # self.model.save('models/Model_Embedding_myTokenizer_'+ str(epoch)+'.keras')
        # self.model.save_weights('models/Weights_Model_Embedding_myTokenizer_'+ str(epoch)+'.keras')
        pattern = 'za góram za lasami żył sobie czerwony kapturek, który skakał sobie po polanie i zbierał fioletowe słodkie, pyszne jagody do kosza'
        num_words_to_generate = 100
        max_len = 20
        generated_text = pattern
        for _ in range(num_words_to_generate):
            input_vectors_padded = self._preprocess_input_text(generated_text, max_len)
            prediction = self.model.predict(input_vectors_padded, verbose = 0)[0]
            prediction_index = np.argmax(prediction)
            predicted_word = tokenizer.index_word[prediction_index]
            generated_text += " " + predicted_word

        print()
        print('Epoch:',  (epoch + 1))
        print('Prediction:', generated_text)


class myLSTM():
    def __init__(self, sequence_len, embedding_dim, lstm_out, tokenizer):
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        self.lstm_out = lstm_out
        self.tokenizer = tokenizer


    def load_weights(self, path):
        self.model.load_weights(path)

    def load_model(self, path):
        self.model = load_model(path)
    

    def create_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=len(self.tokenizer.word_index)+1, output_dim=self.embedding_dim, input_length=self.sequence_len))
        model.add(LSTM(self.lstm_out, dropout=0.2, return_sequences=True))
        model.add(LSTM(self.lstm_out, dropout=0.2, return_sequences=True))
        model.add(LSTM(self.lstm_out, dropout=0.2))
        model.add(Dense(self.lstm_out))
        model.add(Dense(self.lstm_out))
        model.add(Dense(len(self.tokenizer.word_index)+1, activation = 'softmax'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy")
        model.summary()
        self.model = model

    
    def preprocess_input_text(self, text, max_len):
        text = text.split()
        sequence = text[-max_len:]
        vector_sequences = []
        vector_sequence = []
        for word in sequence:
            vector_sequence.append(word)

        vector_sequences.append(vector_sequence)
        vector_sequences = tokenizer.texts_to_sequences(vector_sequences)
            
        vector_diff = max_len - len(vector_sequences[0])
        if not vector_diff:
            return np.array(vector_sequences, dtype=np.int32)
            
        for _ in range(vector_diff):
            vector_sequences[0].insert(0, randint(0, len(tokenizer.word_index)))

        return np.array(vector_sequences, dtype=np.int32)


    def predict(self):
        pattern = 'dawno dawno temu w odległej krainie żyła sobie dziewczynka ktora zbierała pyszne jagody'
        # pattern = 'dawno dawno temu w odległej krainie żyła sobie księżniczka'
        num_words_to_generate = 100
        max_len = 20
        generated_text = pattern
        for _ in range(num_words_to_generate):
            input_vectors_padded = self.preprocess_input_text(generated_text, max_len)
            prediction = self.model.predict(input_vectors_padded, verbose = 0)[0]
            prediction_index = np.argmax(prediction)
            predicted_word = tokenizer.index_word[prediction_index]
            generated_text += " " + predicted_word

        print()
        print('Prediction:', generated_text)


    def fit(self, sequences, targets, epochs=11, batch_size=1024,):
        self.model.fit(sequences, targets, epochs=11, batch_size=1024, callbacks=[PredictionCallback()])
