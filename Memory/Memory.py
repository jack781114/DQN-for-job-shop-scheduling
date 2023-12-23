# 定義了一個經驗回放儲存區（Memory），
# 它可以儲存訓練樣本，並提供隨機取樣功能，
# 用於從經驗回放儲存區取得一批訓練樣本以提供訓練神經網路模型使用。

import random
from collections import deque

class Memory(object):
    def __init__(self, capacity=8000):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def remember(self, sample):
        self.memory.append(sample)

    def sample(self, n):
        n = min(n, len(self.memory))
        sample_batch = random.sample(self.memory, n)
        return sample_batch
