import numpy as np
import utilities as utl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data, labels = utl.load_dataset()
    
    eigenvectors = utl.lda(data, labels)
    
    components = [1, 40, 60]

    image = []
    test_fig = plt.figure()
    test_fig.suptitle('Reconstructed From LDA')
    for i, component in enumerate(components):
        w = eigenvectors[:component]
        lda = np.dot(data, w.T)
        X_reconstructed = np.dot(lda, w) + (np.mean(data, axis=0))
        im = X_reconstructed[0].reshape(64, 64)
        test_ax = test_fig.add_subplot(1, 3, i + 1)
        test_ax.imshow(im)
        test_ax.set_title('K = ' + str(component))
    test_fig.tight_layout()

    utl.plot_mse(data, eigenvectors)
