import numpy as np

class Scores(object):
    @property
    def T(self):
        # self.avg = self.avg.T
        return self.avg.T
        
    def __init__(self, data):
        self.data = np.array(data, dtype=object)
        self.avg = np.array(list(map(lambda x: np.average(x, axis=0), self.data)))
        
    def set_task(self, task):
        if task == 'cls':
            self.avg = np.around(self.avg-1)
            self.data = np.array([[[k-1 for k in j] for j in i] for i in self.data], dtype=object)
        if task == 'bcls':
            t = 3.5
            self.avg = self.avg > t
            self.data = np.array([[[k>t for k in j] for j in i] for i in self.data], dtype=object)
        
    def __getitem__(self, index):
        if isinstance(index, int):
            if hasattr(self, 'avg'):
                return self.avg[index]
            else:
                return self.data[index]
        elif isinstance(index, (list, np.ndarray)):
            if all(isinstance(i, (bool, np.bool_)) for i in index):
                index = [i for i, v in enumerate(index) if v]
            self.data = np.array([self.data[i] for i in index], dtype=object)
            if hasattr(self, 'avg'):
                self.avg = np.array([self.avg[i] for i in index])
            return self
        else:
            raise TypeError("Invalid index type")
        
    def __len__(self):
        return len(self.data)