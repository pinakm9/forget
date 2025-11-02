import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np 
import matplotlib.pyplot as plt

class DiagGaussian(tfp.distributions.MultivariateNormalDiag):
    """
    Description: Simple Gaussian distribution
    """
    def __init__(self, loc, scale):
        super().__init__(loc=loc, scale_diag=scale)
    
    def plot_pdf(self, low, high, resolution=50):
        x = np.linspace(low[0], high[0], num=resolution)
        y = np.linspace(low[1], high[1], num=resolution)
        x_, y_ = np.meshgrid(x, y)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        z = self.prob(np.concatenate([x_.reshape(-1, 1), y_.reshape(-1, 1)], axis=-1)).numpy().reshape(resolution, resolution)
        ax.plot_wireframe(x_, y_, z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()





class GaussianMixture(tfp.distributions.Mixture):
    """
    Description: Simple Gaussian distribution
    """
    def __init__(self, locs, scales, weights):
        Xs = [tfp.distributions.MultivariateNormalDiag(loc=locs[i], scale_diag=scales[i]) for i in range(len(locs))]
        super().__init__(cat=tfp.distributions.Categorical(probs=weights), components=Xs)

    def plot_pdf(self, low, high, resolution=50):
        x = np.linspace(low[0], high[0], num=resolution)
        y = np.linspace(low[1], high[1], num=resolution)
        x_, y_ = np.meshgrid(x, y)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        z = self.prob(np.concatenate([x_.reshape(-1, 1), y_.reshape(-1, 1)], axis=-1)).numpy().reshape(resolution, resolution)
        ax.plot_wireframe(x_, y_, z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    
    




# locs = [[0., -5], [0., 5]]
# scales = [[2., 2.], [2., 2.]]
# weights = [.5, .5]
# rv = GaussianMixture(locs=locs, scales=scales, weights=weights)
# rv.plot_pdf([-5,-5], [5, 5])