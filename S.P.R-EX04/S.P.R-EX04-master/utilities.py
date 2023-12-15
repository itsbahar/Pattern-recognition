import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn import metrics


def load_data(path):
    data = pd.read_csv(path, sep=" ", header=None)
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    return X, y


def logistic_regression(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    logreg = LogisticRegression(solver='liblinear')
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    return acc


def chi_square(x, y, k):
    chi_2 = SelectKBest(chi2, k=k)
    X_k_best_features = chi_2.fit_transform(x, y)
    print(f'Chi-Square Features For K = {k}: {chi_2.get_feature_names_out()}')
    return X_k_best_features


def rfe_score(x, y, k):
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=k)
    X_k_best_features = rfe.fit_transform(x, y)
    print(f'RFE Features For K = {k}: {rfe.get_feature_names_out()}')
    return X_k_best_features


def univariate(x, y, k):
    sel_f = SelectKBest(mutual_info_classif, k=k)
    X_k_best_features = sel_f.fit_transform(x,y)
    print(f'Univariate Features For K = {k}: {sel_f.get_feature_names_out()}')
    return X_k_best_features


def result(x, y, name):
    k = [5, 10]
    print("dataset:", name)
    df = pd.DataFrame(columns=['K', 'Chi-square', 'RFE', 'Univariate'])
    for i in k:
        X_k_best_features_chi_square = chi_square(x, y, i)
        chi_acc = logistic_regression(X_k_best_features_chi_square, y)
        X_k_best_features_rfe = rfe_score(x, y, i)
        rfe_acc = logistic_regression(X_k_best_features_rfe, y)
        X_k_best_features_univariate = univariate(x, y, i)
        Univariate_acc = logistic_regression(X_k_best_features_univariate, y)
        temp_df = pd.DataFrame([[i, chi_acc, rfe_acc, Univariate_acc]],
                               columns=['K', 'Chi-square', 'RFE', 'Univariate'])
        df = pd.concat([df, temp_df])
    print(f'Original Dataset Accuracy: {logistic_regression(x, y)}')
    return df


def load_dataset():
    olivetti = datasets.fetch_olivetti_faces()
    data = olivetti.data
    labels = olivetti.target
    return data, labels


def calc_mu(x, y):
    classes, counts = np.unique(y, return_counts=True)
    c_class = len(classes)
    _, n_feature = x.shape
    mu = np.zeros((c_class, n_feature))
    for i in classes:
        mu[i] = np.sum(np.array(x[y == i]), axis=0) / counts[i]
    return mu


def cal_sw_sb(x, y):
    _, n_feature = x.shape
    classes, counts = np.unique(y, return_counts=True)
    S_W = np.zeros((n_feature, n_feature))
    S_B = np.zeros((n_feature, n_feature))
    mu = calc_mu(x, y)
    mean_overall = np.mean(x, axis=0)
    for cl in classes:
        m_s = (x[y == cl] - mu[cl])
        S_W += np.dot(m_s.T, m_s)
    for cl in classes:
        mean_diff = (mu[cl] - mean_overall).reshape(n_feature, 1)
        S_B += counts[cl] * np.dot(mean_diff, mean_diff.T)
    return S_W, S_B


def calc_mse(y, y_hat):
    diff = y - y_hat
    mse_pow = np.power(diff, 2)
    mse = np.mean(mse_pow)
    return mse


def lda(X, y):
    scatter_w, scatter_b = cal_sw_sb(X, y)
    eigenvalues, eigenvectors = np.linalg.eigh(np.dot(np.linalg.inv(scatter_w), scatter_b))
    eigenvectors = eigenvectors.T
    index = np.argsort(abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[index]
    return eigenvectors


def zero_mean(data):
    feature_means = np.mean(data, axis=0)
    zero_mean_data = data - feature_means
    return zero_mean_data


def calc_eigen_params(data):
    cov = np.dot(data.T, data)
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    return eigen_values, eigen_vectors.T


def pca(data):
    eigen_values, eigen_vectors = calc_eigen_params(data)
    index = np.argsort(-eigen_values)
    eigen_values = eigen_values[index]
    eigen_vectors = eigen_vectors[index]
    return eigen_values, eigen_vectors


def project(data, vector):
    projection = np.dot(data, vector.T)
    return projection


def plot_dataset(data, target):
    fig = plt.figure()
    fig.suptitle(f'Images Of Person {target}')
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(data[target * 10 + i].reshape(64, 64))
        ax.axis('off')
        ax.set_title(f'Image {target * 10 + i}')
    fig.tight_layout()
    plt.show()


def plot_pca_result(data, labels, eigen_vectors):
    dims = [2, 3]
    fig = plt.figure()
    fig.suptitle('PCA For Olivetti Faces Dataset')
    for i, dim in enumerate(dims):
        vector = eigen_vectors[:dim]
        proj = project(data, vector)
        if dim != 3:
            ax = fig.add_subplot(1, 2, i + 1)
            ax.scatter(proj[:, 0], proj[:, 1], c=labels)
        else:
            ax = fig.add_subplot(1, 2, i + 1, projection='3d')
            ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=labels)
            ax.set_zlabel('PC-3')
        ax.set_title('Using ' + str(dim) + ' Principal Components')
        ax.set(xlabel='PC-1', ylabel='PC-2')
    fig.tight_layout()
    plt.show()


def plot_reconstructed(data, eigen_vectors, target):
    components = [1, 20, 50, 150]
    for component in components:
        vectors = eigen_vectors[:component]
        proj = project(data, vectors)
        recon = np.dot(proj, vectors)
        fig = plt.figure()
        fig.suptitle(f'Reconstructed Images Of Person {target} With K = {component}')
        for i in range(10):
            ax = fig.add_subplot(2, 5, i + 1)
            ax.imshow(recon[target * 10 + i].reshape(64, 64))
            ax.axis('off')
            ax.set_title(f'Image {target * 10 + i}')
        fig.tight_layout()
        plt.show()


def plot_eigen_vectors(eigen_vectors):
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('10 First Eigen Vectors')
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.scatter([j for j in range(len(eigen_vectors[i]))], eigen_vectors[i], marker='.')
        ax.set_title(f'Vector {i + 1}')
        ax.set(xlabel='Feature', ylabel='Value')
    fig.tight_layout()
    plt.show()


def plot_mse(data, eigen_vectors):
    mse_mat = []
    for k in range(300):
        vector = eigen_vectors[:k]
        proj = project(data, vector)
        recon = np.dot(proj, vector)
        mse = calc_mse(data.reshape(data.shape[0] * data.shape[1], 1), recon.reshape(data.shape[0] * data.shape[1], 1))
        mse_mat.append(mse)
    fig = plt.figure()
    fig.suptitle('MSE Between Original & Reconstructed Images')
    ax = fig.add_subplot()
    ax.plot(mse_mat)
    ax.set_xlabel('K')
    ax.set_ylabel('MSE')
    fig.tight_layout()
    plt.show()


def variance_analysis(eigen_values):
    values_sum = sum(eigen_values)
    explained_variance = [(i / values_sum) for i in eigen_values]
    gained_var = 0
    best_pc = -1
    f75, f90, f95 = False, False, False
    for i in range(300):
        gained_var += explained_variance[i]
        plt.scatter(i, gained_var, color='green')
        if i == 1:
            print(f'Variance Reached With First PC: {gained_var}')
        elif i == 5:
            print(f'Variance Reached With First 5 PCs: {gained_var}')
        if gained_var >= 0.75 and not f75:
            print(f'Reached 75% Of Variance With {i + 1} PCs')
            f75 = True
        elif gained_var >= 0.90 and not f90:
            f90 = True
            best_pc = i + 1
        elif gained_var >= 0.95 and not f95:
            print(f'Reached 95% Of Variance With {i + 1} PCs')
            f95 = True
    print(f'In Order To Have Good Reconstruction It\'s Said To Have So Many PCs '
          f'That Reach 90% Of Variance Which Is {best_pc} PCs In This Case')
    plt.show()
