from function import unpickle
from lstm import myLSTM


file_path = 'data/tokenizer_080423.pkl'
tokenizer = unpickle(file_path)

def user_input_string():
    while True:
        user_prompt = input()
        try:
            user_prompt = str(user_prompt)
            break

        except ValueError:
            print("Wprowadziłeś nieprawidłowe dane")
            continue

    return user_prompt


def user_input_int():
    while True:
        user_word_to_generate= input()
        try:
            user_word_to_generate = int(user_word_to_generate)
            break

        except ValueError:
            print("Wprowadziłeś nieprawidłowe dane")
            continue

    return user_word_to_generate


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
    print("Wpisz liczbę słów do wygenerowania przez LSTM: ")
    word_to_generate = user_input_int()
    print("Wpisz prompt: ")
    promts = user_input_string()
    lstm.predict(promts, word_to_generate)


if __name__ == "__main__":
    main()
    