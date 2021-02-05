import nltk
import pickle
import argparse
from collections import Counter
from nltk import word_tokenize

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            print('word not in vocab error!!!!')
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab():
    """Build a simple vocabulary wrapper."""
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<sos>')
    vocab.add_word('<eos>')
    vocab.add_word('<unk>')
    vocab.add_word('<eot>')
    vocab.add_word('<eol>')
    vocab.add_word('<sep>')
    vocab.add_word('<s>')
    for i in range(250):
        vocab.add_word(str(i))
    # Add the words to the vocabulary.
    return vocab

def main(args):
    src_vocab = build_vocab()
    src_vocab_path = args.src_vocab_path
    with open(src_vocab_path, 'wb') as f:
        pickle.dump(src_vocab, f)
    print("Total src vocabulary size: {}".format(len(src_vocab)))
    print("Saved the src vocabulary wrapper to '{}'".format(src_vocab_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_path', type=str, default = './data/corpus.txt')
    parser.add_argument('--src_vocab_path', type=str, default='./vocab/vocab.pkl')
    parser.add_argument('--threshold', type=int, default=0)
    args = parser.parse_args()
    main(args)
