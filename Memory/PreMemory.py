# 定義了一個具有優先權的經驗回放儲存區（preMemory），
# 它與普通經驗回放不同，可以根據樣本的TD誤差來調整樣本的優先順序。
# 這允許在訓練時更專注於那些對神經網路預測誤差大的樣本，以提高效果。

import random
from Memory.sum_tree import SumTree as ST

class preMemory(object):
    e = 0.05

    def __init__(self, capacity=8000, pr_scale=0.5):
        self.capacity = capacity
        self.memory = ST(self.capacity)
        self.pr_scale = pr_scale
        self.max_pr = 0

    # 计算優先级
    def get_priority(self, error):
        return (error + self.e) ** self.pr_scale

    def remember(self, sample, error):
        p = self.get_priority(error)
        self_max = max(self.max_pr, p)
        self.memory.add(self_max, sample)


    def sample(self, n):
        sample_batch = []
        sample_batch_indices = []
        sample_batch_priorities = []
        num_segments = self.memory.total() / n

        for i in range(n):
            left = num_segments * i
            right = num_segments * (i + 1)

            s = random.uniform(left, right)
            idx, pr, data = self.memory.get(s)
            sample_batch.append((idx, data))
            sample_batch_indices.append(idx)
            sample_batch_priorities.append(pr)

        return [sample_batch, sample_batch_indices, sample_batch_priorities]

    def update(self, batch_indices, errors):
        for i in range(len(batch_indices)):
            p = self.get_priority(errors[i])
            self.memory.update(batch_indices[i], p)
