import os
import re
import glob
import copy
import logging
import string
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

from .text_data import TextData
from .score import Scores
from .tokenizer import PeerReadTokenizer

from .parsers.Paper import Paper as ParserPaper
from .parsers.ScienceParseReader import ScienceParseReader

logger = logging.getLogger(__name__)


def clean_text(input, lower=False):
    input = input.strip()
    # input = re.sub("\n([0-9]+( *\n))+", "\n", input)
    input = re.sub("\s([0-9]+[\.,]*[0-9]*)[%∗†]*( ([0-9]+[\.,]*[0-9]*)[%∗†]*)+", "", input)
    input = re.sub("\s[0-9]+( [0-9]+)+", "", input)
    if lower: input = input.lower()
    return input

def split_sentence(input):
    return re.split(r'(?<!\s\.)(?<!\s\.\.\.)(?<!\s\w[\.\:])(?<!\s\d\d[\.\:])(?<!fig\.)(?<!eq\.)(?<!eqn\.)(?<!eqns\.)(?<!\w\.\w\.)(?<!\set\.)(?<!\sal\.)(?<!\sal,\.)(?<!vs\.)(?<=[\.\!\?])\s' , re.sub('\s+', ' ', input))


class PeerRead(object):
    def __init__(self, dataset, all_set=False):
        data_dir = '../../research/PeerRead/data'
        sets = ['train', 'dev', 'test']
        
        self.data = defaultdict(list)
        for set in sets:
            review_dir = os.path.join(data_dir, dataset, set, 'reviews')
            scienceparse_dir = os.path.join(data_dir, dataset, set, 'parsed_pdfs/')
            paper_json_filenames = sorted(glob.glob('{}/*json'.format(review_dir)))
            for paper_json_filename in paper_json_filenames:
                d = {}
                paper = ParserPaper.from_json(paper_json_filename)
                paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID, paper.TITLE, paper.ABSTRACT, scienceparse_dir)
                review_contents = []
                reviews = []
                for review in paper.REVIEWS:
                    review_contents.append(clean_text(review.COMMENTS, lower=True))
                    reviews.append(review)
                
                d['paper_id'] = int(paper.ID)    
                d['paper_content'] = clean_text(paper.SCIENCEPARSE.get_paper_content(), lower=True)
                d['reviews_content'] = review_contents
                d['reviews'] = reviews
                d['scores'] = self.get_aspect_scores(reviews)
                self.data[set].append(d)
                
        if all_set:
            all_data = []
            for _, v in self.data.items():
                all_data += v
            self.data = all_data
                
    def get_aspect_scores(self, reviews):
        scores_list = []
        for review in reviews:
            scores = {}
            for aspect in ['recommendation', 'substance', 'meaningful_comparison', 'soundness_correctness', 'originality', 'clarity', 'impact']:
                aspect = aspect.upper()
                scores[aspect] = int(review.__dict__[aspect]) if review.__dict__[aspect] is not None else None
            scores_list.append(scores)    
        return scores_list
        
        
                
class Dataset(object):
    def __init__(self, data, aspects, concat_reviews=False):
        
        self.tokenized = False
        self.sentence_encoded = False
        self.concat_reviews_flag = concat_reviews
        
        assert len(aspects) == 1
        paper_ids = []
        x_p = []
        x_r = []
        y, y_avg = [], []
        for d in data:
            yall = []
            review_contents = []
            for i, review in enumerate(d['reviews']):
                yone = [np.nan] * len(aspects)
                for aid, aspect in enumerate(aspects):
                    aspect = aspect.upper()
                    if aspect in review.__dict__ and review.__dict__[aspect] is not None:
                        yone[aid] = float(review.__dict__[aspect])
                yall.append(yone)
                review_contents.append(d['reviews_content'][i])
        
            paper_ids.append(d['paper_id'])
            x_p.append(d['paper_content'])
            if self.concat_reviews_flag: x_r.append(' '.join(review_contents))
            else: x_r.append(review_contents)
            y_avg.append(np.average(yall, axis=0))
            y.append(yall)
        paper_ids = np.array(paper_ids)
        x_p = TextData(x_p)
        x_r = TextData(x_r)
        y_avg = np.array(y_avg)
        y = Scores(y)
        self.paper_ids = paper_ids[~np.isnan(y_avg).any(axis=1).flatten()]
        self.x_p = x_p[~np.isnan(y_avg).any(axis=1).flatten()]
        self.x_r = x_r[~np.isnan(y_avg).any(axis=1).flatten()]
        self.y = y[~np.isnan(y_avg).any(axis=1).flatten()]
        assert len(self.x_p) == len(self.y)
        
    def set_task(self, task):
        self.y.set_task(task)
        
    def y_hist(self, name):
        plt.clf()
        sns.distplot(self.y.avg > 3.5, kde=False)
        plt.savefig(name)
        
    def sentence_hist(self, name):
        plt.clf()
        sns.displot(list(map(len, [xx for x in self.x_r for xx in x])))
        plt.savefig(name)
        
    def word_hist(self, name):
        plt.clf()
        sns.displot([n for x in self.x_p for n in list(map(len, x))])
        plt.savefig(name)
        
    def split(self):
        self.x_p.apply(split_sentence)
        self.x_r.apply(split_sentence)
        longer_than = list(map(lambda x: len(x) > 50, self.x_p))
        self.x_p = self.x_p[longer_than]
        self.x_r = self.x_r[longer_than]
        self.y = self.y[longer_than]
        for i, x in enumerate(self.x_p):
            self.x_p[i] = [s for s in x if len(s) > 30 and len(s) < 600]
        return self
        
    def preprocess(self):
        with open('../../research/glove.840B.300d.pkl', 'rb') as fp:
            glove = pickle.load(fp)
            
        latin_similar = "'ÆÐƎƏƐƔĲŊŒẞÞǷȜæðǝəɛɣĳŋœĸſßþƿȝĄƁÇĐƊĘĦĮƘŁØƠŞȘŢȚŦŲƯY̨Ƴąɓçđɗęħįƙłøơşșţțŧųưy̨ƴÁÀÂÄǍĂĀÃÅǺĄÆǼǢƁĆĊĈČÇĎḌĐƊÐÉÈĖÊËĚĔĒĘẸƎƏƐĠĜǦĞĢƔáàâäǎăāãåǻąæǽǣɓćċĉčçďḍđɗðéèėêëěĕēęẹǝəɛġĝǧğģɣĤḤĦIÍÌİÎÏǏĬĪĨĮỊĲĴĶƘĹĻŁĽĿʼNŃN̈ŇÑŅŊÓÒÔÖǑŎŌÕŐỌØǾƠŒĥḥħıíìiîïǐĭīĩįịĳĵķƙĸĺļłľŀŉńn̈ňñņŋóòôöǒŏōõőọøǿơœŔŘŖŚŜŠŞȘṢẞŤŢṬŦÞÚÙÛÜǓŬŪŨŰŮŲỤƯẂẀŴẄǷÝỲŶŸȲỸƳŹŻŽẒŕřŗſśŝšşșṣßťţṭŧþúùûüǔŭūũűůųụưẃẁŵẅƿýỳŷÿȳỹƴźżžẓ"
        white_list = string.ascii_letters + string.digits + latin_similar + ' '
        
        glove_chars = ''.join([c for c in glove if len(c) == 1])
        glove_symbols = ''.join([c for c in glove_chars if not c in white_list])
        
        def build_vocab(sentences, verbose =  True):
            """
            :param sentences: list of list of words
            :return: dictionary of words and their count
            """
            vocab = {}
            for sentence in sentences:
                for word in sentence:
                    try:
                        vocab[word] += 1
                    except KeyError:
                        vocab[word] = 1
            return vocab

        if isinstance(self.x_p[0], str): jigsaw_chars = build_vocab(self.x_p)
        else: jigsaw_chars = build_vocab([s for d in self.x_p for s in d])
        jigsaw_symbols = ''.join([c for c in jigsaw_chars if not c in white_list])

        symbols_to_delete = ''.join([c for c in jigsaw_symbols if c not in glove_symbols])
        symbols_to_isolate = ''.join([c for c in jigsaw_symbols if c in glove_symbols])

        isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}
        remove_dict = {ord(c):f'' for c in symbols_to_delete}
        
        def handle_punctuation(x):
            x = x.translate(remove_dict)
            x = x.translate(isolate_dict)
            return x
        
        self.x_p.apply(handle_punctuation)
        self.x_r.apply(handle_punctuation)
        return self
    
    def unsplit(self):
        self.x_p.apply_outer(' '.join)
        self.x_r.apply_outer(' '.join)
        
    def tokenized_dist(self, tokenizer, path):
        try:
            os.makedirs(path)
        except:
            pass
        tokenized_p = tokenizer(self.x_p.tolist())
        plt.clf()
        ax = sns.displot(list(map(len, tokenized_p['input_ids'])))
        ax.set(xlabel='# of WordPiece Tokens')
        plt.savefig(os.path.join(path, 'paper.png'), bbox_inches="tight")
        if isinstance(self.x_r[0], str): tokenized_r = tokenizer(self.x_r.tolist())
        else: tokenized_r = tokenizer([r for rs in self.x_r.tolist() for r in rs])
        plt.clf()
        ax = sns.displot(list(map(len, tokenized_r['input_ids'])))
        ax.set(xlabel='# of WordPiece Tokens')
        plt.savefig(os.path.join(path, 'review.png'), bbox_inches="tight")
            
    def tokenize(self, tokenizer, max_length=512, review_length=None):
        def tokenize_fn(inp):
            tokenized = tokenizer(inp, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
            for key in tokenized.keys(): tokenized[key] = tokenized[key].squeeze()
            return tokenized
        self.x_p.apply(tokenize_fn)
        if review_length is not None:
            self.x_r.apply(tokenize_fn)
        self.tokenized = True
        
    def sentence_encode(self, encoder, paper_length=250, review_length=None):
        if review_length is None: review_length = paper_length
        def pad_sentence(inp, pad_lenght):
            padding = [''] * (pad_lenght - len(inp))
            padded = inp + padding
            padded = padded[:pad_lenght]
            mask = [1] * len(inp)
            mask += [0] * (pad_lenght - len(inp))
            mask = mask[:pad_lenght]
            mask = np.array(mask)
            return padded, mask
        def encode_fn(inp, pad_lenght):
            padded, mask = pad_sentence(inp, pad_lenght)
            encoded = encoder.encode(padded, show_progress_bar=False)
            return encoded, mask
        def encode_p(inp):
            return encode_fn(inp, paper_length)
        def encode_r(inp):
            return encode_fn(inp, review_length)
        self.x_p.apply_outer(encode_p)
        self.x_r.apply_outer(encode_r)
        self.sentence_encoded = True
        
    def iloc(self, indices):
        new_data = copy.deepcopy(self)
        new_data.paper_ids = new_data.paper_ids[indices]
        new_data.x_p = new_data.x_p[indices]
        new_data.x_r = new_data.x_r[indices]
        new_data.y = new_data.y[indices]
        return new_data
    
    def split_reviews(self):
        _x_p, _x_r, _y = [], [], []
        for x_p, x_r, y in zip(self.x_p, self.x_r, self.y.data):
            for r, s in zip(x_r, y):
                _x_p.append(x_p)
                _x_r.append(r)
                _y.append(s)
        self.x_p = _x_p
        self.x_r = _x_r
        self.y = np.array(_y)
        
    def get_data_dict(self):
        data_dict = {
            'paper': self.x_p.data,
            'reviews': self.x_r.data,
            'labels': self.y.avg,
        }
        return data_dict
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        if self.tokenized:
            return {
                'p_input_ids': self.x_p[index]['input_ids'],
                'p_mask': self.x_p[index]['attention_mask'],
                # 'r_input_ids': self.x_r[index]['input_ids'].squeeze(0),
                # 'r_mask': self.x_r[index]['attention_mask'].squeeze(0),
                'labels': self.y[index]
            }
        elif self.sentence_encoded:
            return {
                'p_input': self.x_p[index][0],
                'p_mask': self.x_p[index][1],
                'r_input': self.x_r[index][0],
                'r_mask': self.x_r[index][1],
                'label': self.y[index]
            }
        else:
            return {
                'paper': self.x_p[index],
                'review': self.x_r[index],
                'scores': self.y[index]
            }
    

def original_split(dataset, aspects, tokenizer=None, max_length=1000, review_length=200, task='reg', review_vocab=False):
    dataset = PeerRead(dataset)
    train_dataset = Dataset(dataset.data['train'], aspects)
    dev_dataset = Dataset(dataset.data['dev'], aspects)
    test_dataset = Dataset(dataset.data['test'], aspects)
    
    # train_dataset.y_hist('train.png')
    # dev_dataset.y_hist('dev.png')
    # test_dataset.y_hist('test.png')
 
    if task == 'cls':
        train_dataset.y2cls()
        dev_dataset.y2cls()
        test_dataset.y2cls()
    
    # if tokenizer is None:tokenizer = PeerReadTokenizer(train_dataset, review_vocab=review_vocab)
    # train_dataset.preprocess(tokenizer, max_length, review_length)
    # dev_dataset.preprocess(tokenizer, max_length, review_length)
    # test_dataset.preprocess(tokenizer, max_length, review_length)
    
    logger.info('Train: {}'.format(len(train_dataset)))
    logger.info('Dev: {}'.format(len(dev_dataset)))
    logger.info('Test: {}'.format(len(test_dataset)))
    
    return (train_dataset, dev_dataset, test_dataset), tokenizer.vocab


def all_sets(dataset, aspects, task='reg', concat_reviews=False, split_sentence=False):
    dataset = PeerRead(dataset)
    data = dataset.data['train'] + dataset.data['dev'] + dataset.data['test']
    dataset = Dataset(data, aspects, concat_reviews)
    # dataset.y_hist('score.png')
    if split_sentence: dataset.split().preprocess()
    else: dataset.split().preprocess().unsplit()
    dataset.set_task(task)
    return dataset
                        

def kfold(dataset, train_idx, test_idx, 
          tokenizer=None, sentence_encoder=None,
          paper_length=512, review_length=None):
    train_dataset = dataset.iloc(train_idx)
    test_dataset = dataset.iloc(test_idx)
    vocab = None
    
    if tokenizer is not None and sentence_encoder is not None:
        raise Exception("Cannot use tokenizer and sentence encoder at the same time.")
    
    if tokenizer is None: 
        tokenizer = PeerReadTokenizer(train_dataset)
        
    train_dataset.tokenize(tokenizer, paper_length, review_length)
    test_dataset.tokenize(tokenizer, paper_length, review_length)
    vocab = tokenizer.vocab
        
    if sentence_encoder is not None:
        train_dataset.sentence_encode(sentence_encoder, paper_length, review_length)
        test_dataset.sentence_encode(sentence_encoder, paper_length, review_length)
        
    if dataset.concat_reviews_flag:
        raise NotImplementedError
        train_dataset.split_reviews()
        test_dataset.split_reviews()
    
    logger.info('Train: {}'.format(len(train_dataset)))
    logger.info('Test: {}'.format(len(test_dataset)))
    
    return (train_dataset, test_dataset), vocab
    
    
    
    
