from typing import Dict
import csv

import torch

def read_data_from_file(path, num_rows=None):
    data = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, quotechar='"', doublequote=True, escapechar='\\')
        for i, row in enumerate(reader):
            if num_rows is not None and i >= num_rows:
                break
            data.append((row[1].split(), row[0]))  # text, label
    return data


def make_word_dictionary(data, unk_threshold: int = 0, max_ngrams: int = 1) -> Dict[str, int]:
    '''
    Makes a dictionary of words given a list of tokenized sentences.
    :param data: List of (sentence, label) tuples
    :param unk_threshold: All words below this count threshold are excluded from dictionary and replaced with UNK
    :return: A dictionary of string keys and index values
    '''
    # First count the frequency of each distinct ngram
    ngram_frequency = {}

    # all ngrams of same size are collected first before higher ngrams are counted (this way assertions still pass)
    for n in range(0, max_ngrams): # ngram size
        for sent, _ in data:    # sentences
            for i in range(0, len(sent) - n): # ngrams of size n in sentence
                for j in range(0, n + 1): # words per ngram
                    ngram = sent[i+j] if j < 1 else ngram + " " + sent[i+j]
                if ngram not in ngram_frequency:
                    ngram_frequency[ngram] = 0
                ngram_frequency[ngram] += 1

        # Assign indices to each distinct ngram
        word_to_ix = {'UNK': 0}
        for word, freq in ngram_frequency.items():
            if freq > unk_threshold:  # only add ngrams that are above threshold
                word_to_ix[word] = len(word_to_ix)

    # Print some info on dictionary size
    print(f"At unk_threshold={unk_threshold} and using max. {max_ngrams}-grams, the dictionary contains {len(word_to_ix)} ngrams")

    return word_to_ix


def make_label_dictionary(data) -> Dict[str, int]:
    '''
    Make a dictionary of labels.
    :param data: List of (sentence, label) tuples
    :return: A dictionary of string keys and index values
    '''
    label_to_ix = {}
    for _, label in data:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)
    return label_to_ix


def make_label_vector(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

def get_index_vector(sentence, word_to_ix, max_ngrams = 1):

    indices = []
    for n in range(0, max_ngrams): # ngram size
        for i in range(0, len(sentence) - n):  # ngrams of size n in sentence
            for j in range(0, n + 1):  # words per ngram
                ngram = sentence[i + j] if j < 1 else ngram + " " + sentence[i + j]
            idx = word_to_ix[ngram] if ngram in word_to_ix else word_to_ix["UNK"]
            indices.append(idx)

    onehot_vectors = torch.tensor([indices])

    return onehot_vectors
