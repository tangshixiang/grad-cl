from pymanopt.manifolds import Stiefel
import autograd.numpy as np
from pymanopt import Problem
from pymanopt.solvers import TrustRegions

n = 100
k = 10
A = np.zeros((n, n))
manifold = Stiefel(n, n)


def cost(Y):
    S = np.dot(Y, Y.T)
    delta = .5
    return np.sum(np.sqrt((S - A)**2 + delta**2) - delta)


problem = Problem(manifold=manifold, cost=cost)
solver = TrustRegions()

Y = solver.solve(problem)
import pdb
pdb.set_trace()
S = np.dot(Y, Y.T)
