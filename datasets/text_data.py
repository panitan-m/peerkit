import numpy as np
from transformers.tokenization_utils_base import BatchEncoding

class TextData(object):
    def __init__(self, data):
        self.data = np.array(data, dtype=object)
        
    def __call__(self):
        return self.data
    
    def __len__(self):
        return (len(self.data))
    
    def __setitem__(self, index, value):
        self.data[index] = value
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data[index]
        elif isinstance(index, (list, np.ndarray)):
            if all(isinstance(i, (bool, np.bool_)) for i in index):
                index = [i for i, v in enumerate(index) if v]
            self.data = [self.data[i] for i in index]
            return self
        else:
            raise TypeError("Invalid index type")
    
    def apply(self, function):
        if isinstance(self.data[0], str):
            data = list(map(lambda x: function(x), self.data))
        elif isinstance(self.data[0][0], str):
            data = [list(map(lambda x: function(x), d)) for d in self.data]
        elif isinstance(self.data[0][0][0], str):
            data = [[list(map(lambda x: function(x), r)) for r in d] for d in self.data]
        else:
            raise
        self.data = np.array(data, dtype=object) if not isinstance(data[0], BatchEncoding) else data
        
        
    def apply_outer(self, function):
        if isinstance(self.data[0][0], str):
            data = list(map(lambda x: function(x), self.data))
        elif isinstance(self.data[0][0][0], str):
            data = [list(map(lambda x: function(x), d)) for d in self.data]
        else:
            raise
        self.data = np.array(data, dtype=object) if not isinstance(data[0], tuple) else data
        
    def tolist(self):
        if isinstance(self.data, np.ndarray):
            return self.data.tolist()
        else:
            return self.data