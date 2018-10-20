import numpy as np
import pandas as pd

def _vectorize_data(data, seq_length):

    unique_graphemes = set()
    unique_phonemes = set()

    for word, transcription in zip(data[:, 0], data[:, 1]):
        unique_graphemes |= set(word)
        unique_phonemes |= set(transcription.split())

    grapheme_codes = {grapheme:i + 1 for i, grapheme in enumerate(unique_graphemes)}
    phoneme_codes = {phoneme:i + 1 for i, phoneme in enumerate(unique_phonemes)}

    encoded_words = np.zeros((len(data), seq_length), dtype=int)
    encoded_transcriptions = np.zeros((len(data), seq_length), dtype=int)

    for index, (word, transcription) in enumerate(zip(data[:, 0], data[:, 1])):

        encoded_word = [grapheme_codes[grapheme] for grapheme in word]
        encoded_transcription = [phoneme_codes[phoneme] for phoneme in transcription.split()]

        encoded_words[index, :len(encoded_word)] = encoded_word
        encoded_transcriptions[index, :len(encoded_transcription)] = encoded_transcription

    return grapheme_codes, phoneme_codes, encoded_words, encoded_transcriptions

def load_data():

    np.random.seed(42)
    
    data = np.array(pd.read_csv('train.csv', delimiter=',', index_col='Id'), dtype=str)

    seq_length = 40

    grapheme_codes, phoneme_codes, X, y = _vectorize_data(data, seq_length)

    permutation = np.random.permutation(len(X))

    X = X[permutation]
    y = y[permutation]

    train_X = X[:90000]
    train_y = y[:90000]
    val_X = X[90000:95000]
    val_y = y[90000:95000]
    test_X = X[95000:]
    test_y = y[95000:]

    return grapheme_codes, phoneme_codes, train_X, train_y, val_X, val_y, test_X, test_y