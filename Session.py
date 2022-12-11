import numpy as np
from collections import namedtuple

class Variable:
    def __init__(self):
        self.data = None
        self.col = None
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.lo = None
        self.hi = None

    def update(self, data):
        # Update the given variable's attributes, assuming the col it represents has changed
        # ONLY CALLED ON CHANGE OF DATA.

        # Can only be computed once the col attribute is given for both, since we compute it from the
        # data[col] data.

        self.data = data[self.col]
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.min = np.amin(self.data)
        self.max = np.amax(self.data)

    def print(self):
        print("col: ", self.col)
        print("mean: ", self.mean)
        print("std: ", self.std)
        print("min: ", self.min)
        print("max: ", self.max)
        print("lo: ", self.lo)
        print("hi: ", self.hi)


    def interval(self, i):
        #self.update(var)
        #print(var.std)
        return [float(self.mean-i*self.std), float(self.mean+i*self.std)]


class Session:
    def __init__(self, data):
        """
        Class for managing session variables and functions throughout usage of quicksilver,
            including the dataset they filter through and the operations chosen.

        Tried to do this without one, but it got to be a debugging and variable scope nightmare.

        Implementing this to fix that.

        :param data:

        """
        # Init all session variables
        self.data = data

        #var = namedtuple("var", "data col mean std min max lo hi interval", defaults=[None, None, None, None, None, None, None, None, None])
        self.x = Variable()
        self.y = Variable()


        # TODO do we need this?
        #self.graphing = False



    def sync(self):
        # do we need this?
        # sync with the session state. only implement if needed.
        pass



test = Session("asdf")
