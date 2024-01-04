import utilities as utl
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = ['./datasets/blobs.csv', './datasets/circle.csv', './datasets/elliptical.csv',
            './datasets/moon.csv', './datasets/tsnv.csv']
    for i in range(len(path)):
        data = utl.load_data(path[i])
        if i == 4:
            X = data.iloc[:, :]
        else:
            X = data.iloc[:, :-1]
        X = utl.normalize(X).values
        WCSS = []
        silhouettes = []
        for K in range(1, 15):
            clusters, means = utl.k_mans(X, K, 300)
            WCSS.append(utl.Wcss(clusters, means, K))
            if K !=1:
               x,y=utl.getlabel(clusters) 
               silhouettes.append(silhouette_score(x,y, metric = 'euclidean')) 
        utl.plot_elbow(WCSS)
        plt.plot(range(2,15),silhouettes)
        plt.show()
