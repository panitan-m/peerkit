import logging
import numpy as np
import pickle
from collections import OrderedDict, Counter
from nltk.tokenize.treebank import TreebankWordTokenizer

logger = logging.getLogger(__name__)

class PeerReadTokenizer(object):
    
    def __init__(self, dataset, max_vocab_size=35000, review_vocab=False):
        logger.info('Building vocab ...')
        words = []
        self.tokenizer = TreebankWordTokenizer()
        for d in dataset.x_p.data:
            if isinstance(d, str): words += self.tokenizer.tokenize(d)
            elif isinstance(d, list):
                for s in d: words += self.tokenizer.tokenize(s)
            else:
                raise ValueError
            if review_vocab:
                raise NotImplementedError
                if isinstance(d[1], str): words += self.tokenizer.tokenize(d[0])
                else:
                    for s in d[1]: words += self.tokenizer.tokenize(s)
        logger.info('Total words in corpus: {}'.format(len(words)))
    
        vocab = OrderedDict()
        word_counter = Counter(words)
        vocab['PAD'] = 0
        vocab['UNK'] = 1
        self.pad_token_id = vocab['PAD']
        for w, _ in word_counter.most_common():
            if max_vocab_size:
                if len(vocab) >= max_vocab_size:
                    break
            if len(w) and w not in vocab:
                vocab[w] = len(vocab)
        self.vocab = vocab
        self.vocab_inv = {int(i):v for v,i in vocab.items()}
        self.vocab_size = len(vocab)
        logger.info('Total vocab of size: {}'.format(len(vocab)))
        
    def __call__(self, input, **kwds):
        tokenized = [self.vocab[w] if w in self.vocab else 1 for w in self.tokenizer.tokenize(input)]
        padded = self.pad_sentence(tokenized, kwds['max_length'], self.pad_token_id)
        output = {}
        output["input_ids"] = np.array(padded)
        output["attention_mask"] = (np.array(padded) != self.pad_token_id).astype(int)
        return output
            
    def pad_sentence(self, token_list, pad_length, pad_id, reverse=False):
        if reverse:
            token_list = token_list[::-1]
            padding = [pad_id] * (pad_length - len(token_list))
            padded_list = padding + token_list
        else:
            padding = [pad_id] * (pad_length - len(token_list))
            padded_list = token_list + padding
        return padded_list[:pad_length]
    
    
def load_embeddings(vocab, file):
    with open(file, 'rb') as f:
        glove_embedding = pickle.load(f)
    embedding_size = len(glove_embedding['the'])
    
    embedding_var = np.random.normal(0.0, 0.01,[len(vocab), embedding_size] )
    no_embeddings = 0
    
    for word, wid in vocab.items():
        try:
            embedding_var[wid,:] = glove_embedding[word]
        except KeyError:
            no_embeddings +=1
        continue
    logger.info("num embeddings with no value: {} / {}".format(no_embeddings, len(vocab)))
    return np.array(embedding_var, dtype=np.float32)
