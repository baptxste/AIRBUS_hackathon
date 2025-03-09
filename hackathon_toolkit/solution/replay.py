from collections import deque
class ReplayBuffer:
    def __init__(self):
        self.memory = deque(maxlen=100000)

    def store(self, data):
        self.memory.append(data)

    def get_data(self):
        return self.memory