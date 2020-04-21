import pandas as pd
import re
import numpy as np
import torch
import torch.nn as nn

from typing import List
from torch.utils.data import Dataset, DataLoader, random_split
from os import path, listdir

class StoriesDataset(Dataset):
    def __init__(self, data_path, word_embedding):
        self.data_path = data_path

        if path.exists(data_path):
            self.dataset = pd.read_csv(data_path)

        self.word_embedding = word_embedding

        return

    def generate_sequences(self, stories_path: str, story_header: str, n_stories: int, seq_len: int):
        stories = pd.read_csv(stories_path, nrows=n_stories)[story_header]

        stories = stories.map(lambda x: re.sub(r"\W+", ' ', x).strip("  "))  # remove all triple spaces and all non words
        stories = stories.str.lower().str.replace('_',"").str.split()

        x = []
        y = []
        seqdata = pd.DataFrame(columns=['sequence', 'target'])
        for index, story in stories.iteritems():
            for i in range(0, len(story) - seq_len, 1):
                rs = story[i:i + seq_len]
                x.append(' '.join(rs))
                y.append(story[i + seq_len])

        seqdata['sequence'] = x
        seqdata['target'] = y
        seqdata.to_csv(self.data_path)

        self.dataset = seqdata

        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        target = self.word_embedding.get_word_vectors([self.dataset['target'][idx]])
        sequence = self.dataset['sequence'][idx].split()
        sequence_vects = self.word_embedding.get_word_vectors(sequence)

        return sequence_vects, target


class WordEmbedding():
    def __init__(self, embeddings_path: str):
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

        self.vectors = np.asarray(vectors)
        self.dictionary = word2idx
        self.dim = self.vectors.shape[1]

        print('Made an embedding containing ', len(words), 'words.')

        return

    def _argmax_top_k(self, x: np.array, k: int)-> (np.array, np.array):
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

    def _similarities(self, word_w: str) -> np.array:
        w = self.dictionary[word_w]
        vectors_t = np.transpose(self.vectors)

        n_num = self.vectors[w] @ vectors_t
        n_denom = np.linalg.norm(self.vectors[w], 2) * np.linalg.norm(vectors_t, 2)
        sims = n_num / n_denom

        return sims.reshape(1, len(self.dictionary))

    def similarity(self, word_i: str, word_j: str) -> float:
        i = self.dictionary[word_i]
        j = self.dictionary[word_j]
        v_i = self.vectors[i] / np.linalg.norm(self.vectors[i], 2)  # a/|a|
        v_j = self.vectors[j] / np.linalg.norm(self.vectors[j], 2)  # b/|b|

        return np.matmul(v_i.reshape(1, -1), v_j.reshape(-1, 1)).item()

    def most_similar(self, word_i: str, k: int)-> List:
        k += 1
        sims = self._similarities(word_i)
        _, topi = self._argmax_top_k(sims, k)
        inv = {v: i for i, v in self.dictionary.items()}
        return [inv[i[0]] for i in topi[1:k]]

    def analogy(self, word_a: str, word_b: str, word_c: str, k: int) -> List:

        v_d = self.vectors[self.dictionary[word_b]] - self.vectors[self.dictionary[word_a]] + self.vectors[self.dictionary[word_c]]
        vectors_t = np.transpose(self.vectors)

        n_t = v_d @ vectors_t
        n_b = np.linalg.norm(v_d, 2) * np.linalg.norm(vectors_t, 2)
        sims = n_t / n_b
        sims = sims.reshape(1, len(self.dictionary))

        _, topi = self._argmax_top_k(sims, k)
        inv = {v: i for i, v in self.dictionary.items()}

        return [inv[i[0]] for i in topi]

    def get_word_vectors(self, text: List)-> np.array:
        emb_dim = self.vectors.shape[1]
        matrix_len = len(text)
        weights_matrix = np.zeros((matrix_len, emb_dim))

        for i, word in enumerate(text):
            try:
                weights_matrix[i] = self.vectors[self.dictionary[word]]
            except KeyError:
                weights_matrix[i] = np.zeros((emb_dim,))

        return weights_matrix


class LSTMWordpred(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers):
        super(LSTMWordpred, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # lstm layers
        self.rnns = nn.ModuleList()
        for i in range(n_layers):
            input_size = self.embedding_dim if i == 0 else self.hidden_dim
            self.rnns.append(nn.LSTM(input_size, self.hidden_dim, 1))

        # linear layer
        self.dense_linear = nn.Linear(self.hidden_dim, self.embedding_dim)

    def forward(self, sequence):

        # run sequences through all lstm layers
        for i in range(len(self.rnns)):
            if i != 0:
                sequence = torch.nn.functional.dropout(sequence, p=0.2, training=True)
            output, hidden = self.rnns[i](sequence)
            sequence = output

        lstm_out = sequence[:, -1, :]

        out = self.dense_linear(lstm_out)

        return out

    def fit(self, dataset, n_epochs, val_split, batch_size, loss_function, opt, device='cpu'):

        # split dataset
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        # make dataloaders
        train_gen = DataLoader(train_set, shuffle=True, num_workers=2, batch_size=batch_size)
        val_gen = DataLoader(val_set, shuffle=True, num_workers=2, batch_size=batch_size)

        train_hist = []
        val_hist = []

        print(f'Fitting model with {n_epochs} epochs over {len(dataset)} sequences.')

        for epoch in range(n_epochs):
            train_loss = 0
            val_loss = 0
            i = 0
            print(f'Starting epoch {epoch+1}')
            for local_seq, local_tar in train_gen:
                self.zero_grad()

                local_seq = torch.squeeze(local_seq.float(), dim=1).to(device)
                local_tar = torch.squeeze(local_tar.float(), dim=1).to(device)

                out = self(local_seq)
                # print(out.shape)

                loss = loss_function(out, local_tar)
                loss.backward()
                opt.step()

                train_loss += loss.item()
                i += 1

            train_loss = train_loss / i
            train_hist.append(train_loss)

            with torch.no_grad():
                i = 0
                for local_seq, local_tar in val_gen:
                    local_seq = torch.squeeze(local_seq.float(), dim=1).to(device)
                    local_tar = torch.squeeze(local_tar.float(), dim=1).to(device)

                    out = self(local_seq)
                    loss = loss_function(out, local_tar)
                    val_loss += loss.item()
                    i += 1

            val_loss = val_loss / i
            val_hist.append(val_loss)

            print(f'Training loss: {train_loss}, Validation loss: {val_loss}')

        return train_hist, val_hist

    def get_hidden_states(self, sequence: List)-> List:

        # run sequence through all lstm layers
        outputs = []
        for i in range(len(self.rnns)):
            if i != 0:
                sequence = torch.nn.functional.dropout(sequence, p=0.2, training=False)
            output, hidden = self.rnns[i](sequence)
            sequence = output
            outputs.append(output[:, -1, :])

        return outputs


def load_models(model, pretrained_weights_dir):
    models = {}
    for filename in listdir(pretrained_weights_dir):
        if filename.endswith(".pth"):
            model.load_state_dict(torch.load(path.join(pretrained_weights_dir, filename)))
            model.eval()

            modelname = filename[:-4]

            models[modelname] = model

        print(f'Loaded model weights of {filename} into {modelname}')

    return models



