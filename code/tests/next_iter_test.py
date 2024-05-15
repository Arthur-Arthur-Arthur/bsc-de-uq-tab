import numpy as np


class DataLoader():
    def __init__(self):
        self.data = [1, 2, 3, 4, 5]
        self.index = 0

    def __iter__(self):
        self.index = 0
        np.random.shuffle(self.data)
        return self

    def __next__(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration()

loader_train = DataLoader()

iter2 = iter(loader_train)
iter3 = iter(loader_train)

for each1 in iter(loader_train):
    each2 = next(iter2)
    each3 = next(iter3)
    print(each1, each2, each3)