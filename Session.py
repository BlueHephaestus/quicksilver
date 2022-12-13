import numpy as np
from collections import namedtuple

class Variable:
    def __init__(self, name, state):
        self.name = name
        self.state = state
        self.data = None
        self.col = None
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.lo = None
        self.hi = None
        print(self.state)

    def update(self, data):
        # Update the given variable's attributes, assuming the col it represents has changed
        # ONLY CALLED ON CHANGE OF DATA.

        # Can only be computed once the col attribute is given for both, since we compute it from the
        # data[col] data.

        self.data = data[self.col]
        # TODO we need a case here for if data is non-numeric, what to do?
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
        print()

    def __getattribute__(self, item):
        # Get from session state
        #self.item = self.state[item]
        # Update our own value with whatever session state has
        #self.__setattr__(item, self.state[item])
        if item not in ["data", "col", "mean", "std", "min", "max", "lo", "hi"]:
            # don't do this for state, since that's infinite recursion
            return super().__getattribute__(item)
        item = self.name + "_" + item
        if item == "x_lo":
            print("getting x_lo", self.state[item])
        #print(f"{item} = self.state[{item}]")
        #eval(f"{item} = self.state[{item}]")
        self.__setattr__(item, self.state[item])
        return self.state[item]

    def __setattr__(self, key, value):
        if key not in ["data", "col", "mean", "std", "min", "max", "lo", "hi"]:
            # don't do this for state, since that's infinite recursion
            super().__setattr__(key, value)
            return

        key = self.name + "_" + key

        # Set into session state and class state.
        self.state[key] = value
        super().__setattr__(key, value)

    def interval(self, i):
        #self.update(var)
        #print(var.std)
        lo = float(max(self.min, self.mean - i * self.std))
        hi = float(min(self.max, self.mean + i * self.std))
        return [lo,hi]


class Session:
    def __init__(self, data, state):
        """
        Class for managing session variables and functions throughout usage of quicksilver,
            including the dataset they filter through and the operations chosen.

        Tried to do this without one, but it got to be a debugging and variable scope nightmare.

        Implementing this to fix that.

        :param data:

        """
        # Init all session variables
        self.data = data

        self.x = Variable("x", state)
        self.y = Variable("y", state)
        self.scatter_enable = False


        # TODO do we need this?
        #self.graphing = False



    def sync(self):
        # do we need this?
        # sync with the session state. only implement if needed.
        pass



