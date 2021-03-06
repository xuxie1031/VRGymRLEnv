import copy
import numpy as np

class Replay:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0

    
    def feed(self, experience):
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos+1) % self.memory_size
    

    def feed_batch(self, experience):
        experience = zip(*experience)
        for exp in experience:
            self.feed(exp)


    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[idx] for idx in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        return batch_data


    def sample_segment(self, pos1, pos2):
        if pos1 >= pos2:
            return None

        sampled_indices = range(pos1, pos2)
        sampled_data = [self.data[idx] for idx in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        return batch_data


    def size(self):
        return len(self.data) 


    def reset(self):
        del self.data[:]
        self.pos = 0


    def set_params(self, memory_size, batch_size, data, pos):
        self.memory_size, self.batch_size, self.pos = memory_size, batch_size, pos
        self.data = copy.copy(data)


    def get_params(self):
        return self.memory_size, self.batch_size, self.data, self.pos