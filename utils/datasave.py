class DataSaver(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.last = 0
        self.max = -1000
        self.argmax = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.last = 0

    def update(self, val, epoch=-100):
        self.last = self.val
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
            self.argmax = epoch
