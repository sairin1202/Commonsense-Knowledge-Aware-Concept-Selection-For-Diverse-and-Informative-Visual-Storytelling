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
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(file, threshold, max_size):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    with open(file, 'r') as f:
        data = f.readlines()
    data = [d[:-1] for d in data]

    for i, d in enumerate(data):
        tokens = list(word_tokenize(d.lower()))
        # tokens = d.lower().split(" ")
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the texts.".format(i+1, len(data)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    words = words[:max_size]
    with open('vocab/cnter.pkl','wb') as f:
        pickle.dump(counter, f)
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<sos>')
    vocab.add_word('<eos>')
    vocab.add_word('<unk>')
    vocab.add_word('<eot>')
    vocab.add_word('<eol>')
    vocab.add_word('<s>')
    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    src_vocab = build_vocab(file=args.src_data_path, threshold=args.threshold, max_size=args.max_size)
    src_vocab_path = args.src_vocab_path
    with open(src_vocab_path, 'wb') as f:
        pickle.dump(src_vocab, f)
    print("Total src vocabulary size: {}".format(len(src_vocab)))
    print("Saved the src vocabulary wrapper to '{}'".format(src_vocab_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_path', type=str, default = './data/corpus.txt')
    parser.add_argument('--src_vocab_path', type=str, default='./vocab/vocab.pkl')
    parser.add_argument('--threshold', type=int, default=5)
    parser.add_argument('--max_size', type=int, default=50000)
    args = parser.parse_args()
    main(args)
