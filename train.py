from function import unpickle
from lstm import myLSTM


file_path = 'data/tokenizer_080423.pkl'
tokenizer = unpickle(file_path)
file_path = 'data/targets_080423.pkl'
targets = unpickle(file_path)
file_path = 'data/sequences_080423.pkl'
sequences = unpickle(file_path)

def train():
    sequence_len = 20
    embedding_dim = 200
    lstm_out = 512
    lstm = myLSTM(
        sequence_len = sequence_len, 
        embedding_dim = embedding_dim, 
        lstm_out = lstm_out,
        tokenizer = tokenizer)
    lstm.create_model()
    lstm.fit(sequences=sequences, targets=targets, epochs=11, batch_size=1024)


if __name__ == "__main__":
    train()