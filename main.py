from function import unpickle
from lstm import myLSTM


file_path = 'data/tokenizer_080423.pkl'
tokenizer = unpickle(file_path)
def main():
    sequence_len = 20
    embedding_dim = 200
    lstm_out = 512
    lstm = myLSTM(
        sequence_len = sequence_len, 
        embedding_dim = embedding_dim, 
        lstm_out = lstm_out,
        tokenizer = tokenizer)
    path = "models/Model_Embedding_myTokenizer.keras"
    lstm.load_model(path)
    path = "models/weights_Model_Embedding_myTokenizer.keras"
    lstm.load_weights(path)
    lstm.predict()


if __name__ == "__main__":
    main()