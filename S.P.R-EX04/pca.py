import utilities as utl


if __name__ == '__main__':
    data, labels = utl.load_dataset()

    data = utl.zero_mean(data)

    values, vectors = utl.pca(data)

    utl.plot_dataset(data, 20)

    utl.plot_pca_result(data, labels, vectors)

    utl.plot_reconstructed(data, vectors, 20)

    utl.plot_eigen_vectors(vectors[:10])

    utl.plot_mse(data, vectors)

    utl.variance_analysis(values)
