import random

class RandomAgent:
    def __init__(self, K):
        self.K = K

    def choose(self):
        return random.randrange(self.K)