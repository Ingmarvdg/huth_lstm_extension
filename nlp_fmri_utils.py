import pandas as pd

import re
from typing import Dict, List
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from keras.layers import LSTM, Dense, Masking
from keras.models import Sequential
from keras import backend as K

def pre_process_stories(data_path: str, n_stories: int, p_size_lim: int, content_row_name: str) -> pd.DataFrame:
    processed_data = pd.DataFrame()

    story_data = pd.read_csv(data_path, delimiter=',', nrows = n_stories)

    for index, row in story_data.iterrows():
        story = row[content_row_name]  # get the content of the story
        split_story = pd.DataFrame(story.split("\n\n\n"))  # split the story by paragraph

        split_story[0] = split_story[0].map(lambda x: x.strip('\n').lower())  # remove all the newlines in the text
        split_story[0] = split_story[0].map(lambda x: re.sub(r"\W+", ' ', x).strip("  "))  # remove all triple spaces and all non words
        split_story[0] = split_story[0].str.split()  # split each paragraph up into a list of words
        split_story = split_story[split_story[0].apply(lambda x: len(x) > p_size_lim)]  # remove all paragraphs that are too short

    return processed_data.append(split_story)


def load_word_embeddings(embeddings_path: str)-> (Dict[str, int], np.array):
    words = []
    i = 0
    word2idx = {}
    vectors = []

    with open(embeddings_path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = i
            i += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = np.asarray(vectors)
    dictionary = word2idx
    print('Made an embedding containing ', len(words), 'words.')

    return dictionary, vectors


def vectorize(text: List, embeddings_matrix: np.array, embeddings_dict: Dict[str,int])-> np.array:
    emb_dim = embeddings_matrix.shape[1]
    matrix_len = len(text)
    weights_matrix = np.zeros((matrix_len, emb_dim))

    for i, word in enumerate(text):
        try:
            weights_matrix[i] = embeddings_matrix[embeddings_dict[word]]
        except KeyError:
            weights_matrix[i] = np.zeros((emb_dim,))

    return weights_matrix


def make_sequences(text: np.array, seq_length: int)-> (List, List):
    x = []
    y = []
    for i in range(0, len(text) - seq_length, 1):
        x.append(text[i:i + seq_length])
        y.append(text[i + seq_length])

    return x, y


def make_sequences_set(text_set: List, seq_length: int, vectors: np.array, vocab: Dict[str, int], max_len: int,
                       padding_val: float) -> (np.array, np.array):
    seqs = []
    targets = []
    for text in text_set:
        text_v = vectorize(text, vectors, vocab)
        text_s, text_t = make_sequences(text_v, seq_length)
        text_s = pad_sequences(text_s, max_len, value=padding_val, padding='pre', dtype='float')
        targets.extend(text_t)
        seqs.extend(text_s)

    x = np.array(seqs)
    y = np.array(targets)

    print("Amount of sequences: ", len(x))

    return x, y


def similarity(word_i: str, word_j: str, vocab: Dict[str,int], vectors: np.array) -> float:
    i = vocab[word_i]
    j = vocab[word_j]
    v_i = vectors[i] / np.linalg.norm(vectors[i], 2)  # a/|a|
    v_j = vectors[j] / np.linalg.norm(vectors[j], 2)  # b/|b|

    return np.matmul(v_i.reshape(1, -1), v_j.reshape(-1, 1)).item()


def similarities(word_w: str, vocab: Dict[str,int], vectors: np.array)-> np.array:
    w = vocab[word_w]
    vectors_t = np.transpose(vectors)

    n_num = vectors[w] @ vectors_t
    n_denom = np.linalg.norm(vectors[w], 2) * np.linalg.norm(vectors_t, 2)
    sims = n_num / n_denom

    return sims.reshape(1, len(vocab))


def most_similar(word_i: str, vocab: Dict[str,int], vectors: np.array, k: int)-> List:
    k = k + 1
    sims = similarities(word_i, vocab, vectors)
    _, topi = argmax_top_k(sims, k)
    inv = {v: i for i, v in vocab.items()}
    return [inv[i[0]] for i in topi[1:k]]


def argmax_top_k(x: np.array, k: int)-> (np.array, np.array):
    copy = x
    retv, reti = [], []
    for repeat in range(k):
        values = np.max(copy)
        indices = np.where(copy == np.max(copy))
        mask = np.arange(x.shape[1]).reshape(1, -1) == indices[1]
        copy[mask] = -float('inf')
        retv.append(values)
        reti.append(indices[1])
    return retv, reti


def analogy(word_a: str, word_b: str, word_c: str, vocab: Dict[str,int], vectors: np.array, k: int) -> List:

    v_d = vectors[vocab[word_b]] - vectors[vocab[word_a]] + vectors[vocab[word_c]]
    vectors_t = np.transpose(vectors)

    n_t = v_d @ vectors_t
    n_b = np.linalg.norm(v_d, 2) * np.linalg.norm(vectors_t, 2)
    sims = n_t / n_b
    sims = sims.reshape(1, len(vocab))

    _, topi = argmax_top_k(sims, k)
    inv = {v: i for i, v in vocab.items()}

    return [inv[i[0]] for i in topi]

def build_context_rep_function(story_data: List, seq_length: int, vectors: np.array, vocab: Dict[str,int]) -> K.function:
    max_seq_len = 20
    PAD = 0.0
    sequences, targets = make_sequences_set(story_data, seq_length, vectors, vocab, max_seq_len, PAD)
    lstm_layer_size = 50
    model = Sequential()
    model.add(Masking(mask_value=PAD, input_shape=(max_seq_len, 50)))
    model.add(LSTM(lstm_layer_size,
                   dropout=0.2,
                   input_shape=(max_seq_len, 50), recurrent_dropout=0.2, return_sequences=True, name='lstm1'))

    model.add(LSTM(lstm_layer_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, name='lstm2'))
    model.add(LSTM(lstm_layer_size, name='lstm3'))
    model.add(Dense(targets.shape[1], activation='linear'))
    model.compile(loss='cosine_similarity', optimizer='adam', metrics=['cosine_similarity'])

    n_epochs = 50
    model.fit(sequences,
                targets,
                epochs=n_epochs, batch_size=sequences.shape[0], verbose=1, validation_split=0.01)

    return K.function([model.layers[0].input],
                        [model.layers[3].output])