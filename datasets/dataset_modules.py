import random
import numpy as np
from torch.utils.data import IterableDataset


class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size
        
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass
    
    
class RankingDataset(IterableDataset):
    def __init__(self, dataset, triplet=False):
        self.data = dataset
        self.triplet = False
        
    def __len__(self):
        if self.triplet:
            raise NotImplementedError
        else:
            n = len(self.data)
            return int(n * (n-1) / 2)
        
    def __iter__(self):
        if self.triplet:
            raise NotImplementedError
        else:
            for i in range(len(self.data)):
                for j in range(i+1, len(self.data)):
                    yield {
                        'p0_input_ids': self.data[i]['p_input_ids'],
                        'p0_mask': self.data[i]['p_mask'],
                        'r0_input_ids': self.data[i]['r_input_ids'],
                        'r0_mask': self.data[i]['r_mask'],
                        'p1_input_ids': self.data[j]['p_input_ids'],
                        'p1_mask': self.data[j]['p_input_ids'],
                        'r1_input_ids': self.data[j]['r_input_ids'],
                        'r1_mask': self.data[j]['r_mask'],
                        'label': np.array([1 if x else -1 for x in self.data[i]['label'] == self.data[j]['label']])
                    }
        