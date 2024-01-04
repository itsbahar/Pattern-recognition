import numpy as np
import utilities as utl
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


if __name__ == '__main__':
    path = ['./datasets/blobs.csv', './datasets/circle.csv', './datasets/elliptical.csv',
            './datasets/moon.csv', './datasets/tsnv.csv']
    components = [1, 5, 10]
    iterations = 65
    for i in range(len(path)):
        data = utl.load_data(path[i])
        if i == 4:
            x = data.iloc[:, :]
        else:
            x = data.iloc[:, :-1]
        x = utl.normalize(x).values
        for n_component in components:
            phi = [1 / n_component for _ in range(n_component)]
            mu, sigma = [], []
            clusters = np.array_split(x, n_component)
            for cluster in clusters:
                mu.append(np.mean(cluster, axis=0))
                sigma.append(np.cov(cluster, rowvar=False))
            for _ in range(iterations):
                ev = utl.e_step(x, phi, mu, sigma, n_component)
                phi, mu, sigma = utl.m_step(x, ev)
            pdfs = [multivariate_normal.pdf(x, mu[n], sigma[n], allow_singular=True) for n in range(n_component)]
            cluster_number = np.argmax(pdfs, axis=0)
            plt.scatter(x[:, 0], x[:, 1], marker='.', c=cluster_number)
            plt.title(f'GMM For Dataset {i + 1} With {n_component} Components')
            plt.show()
