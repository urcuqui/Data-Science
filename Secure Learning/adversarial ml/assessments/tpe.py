import numpy as np
from scipy.stats import norm

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def insert(self, data):
        if data < self.data:
            if self.left is None:
                self.left = Node(data)
            else:
                self.left.insert(data)
        elif data > self.data:
            if self.right is None:
                self.right = Node(data)
            else:
                self.right.insert(data)

    def in_order_traversal(self):
        res = []
        if self.left is not None:
            res = self.left.in_order_traversal()
        res.append(self.data)
        if self.right is not None:
            res = res + self.right.in_order_traversal()
        return res

class TreeParzenEstimator:
    def __init__(self, bounds):
        self.bounds = bounds
        self.X = []
        self.Y = []
        self.trees = {}

    def _split(self, data, split):
        return [x for x in data if x < split], [x for x in data if x >= split]

    def _prior(self, x):
        a, b = self.bounds
        return 1 / (b - a)

    def _likelihood(self, x, tree, split):
        l_data, r_data = self._split(tree.in_order_traversal(), split)
        l = len(l_data)
        r = len(r_data)
        total = l + r
        if x < split:
            return l / total * norm.pdf(x, np.mean(l_data), np.std(l_data))
        else:
            return r / total * norm.pdf(x, np.mean(r_data), np.std(r_data))

    def suggest(self, n_suggestions=1):
        if len(self.X) < 2:
            return np.random.uniform(self.bounds[0], self.bounds[1], size=n_suggestions)
        else:
            x = np.linspace(self.bounds[0], self.bounds[1], 1000)
            y = [self._likelihood(i, self.trees[np.argmin(self.Y)], np.median(self.X)) / self._prior(i) for i in x]
            return x[np.argmax(y)]

    def observe(self, x, y):
        self.X.extend(x)
        self.Y.extend(y)
        for xi, yi in zip(x, y):
            if yi not in self.trees:
                self.trees[yi] = Node(xi)
            else:
                self.trees[yi].insert(xi)

                
# tpe = TreeParzenEstimator(bounds=(0, 10))

# for _ in range(20):
#     x = tpe.suggest()
#     y = some_function(x)
#     tpe.observe([x], [y])
