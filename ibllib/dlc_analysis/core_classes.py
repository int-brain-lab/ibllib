import numpy as np

class TrackedPoint(object):
    # instance attribute
    def __init__(self, name, x, y, likelihood):
        # attributes
        self.name = name
        self.x = x
        self.y = y
        self.likelihood = likelihood

    @property
    def get_name(self):
        return self.name

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_likelihood(self):
        return self.likelihood