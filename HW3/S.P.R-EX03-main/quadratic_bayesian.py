import numpy as np
import utilities as utl
import matplotlib.pyplot as plt
import discriminative_analysis as da
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Shared Info Between Datasets
    dataset_classes = 3
    dataset_class_size = 500

    # Dataset 1 Data Preparation
    dataset_1_mu = [[3, 6], [5, 4], [6, 6]]
    dataset_1_sigma = [[[1.5, 0], [0, 1.5]], [[2, 0], [0, 2]], [[1, 0], [0, 1]]]

    dataset_1 = utl.generate_dataset(dataset_1_mu, dataset_1_sigma, dataset_classes, dataset_class_size)

    train_1, test_1 = train_test_split(dataset_1, test_size=0.2)

    # Train Phase Of Dataset 1
    dataset_1_phi, dataset_1_mu, dataset_1_sigma = utl.calc_params(train_1)

    # Test Phase Of Dataset 1
    train_1_pred = utl.bayesian_prediction(train_1, dataset_1_phi, dataset_1_mu, dataset_1_sigma)
    train_1_conf_mat, train_1_score_mat = utl.confusion_score_matrix(train_1.y, train_1_pred)
    train_1_acc = utl.calc_accuracy(np.array(train_1.y), train_1_pred)

    test_1_pred = utl.bayesian_prediction(test_1, dataset_1_phi, dataset_1_mu, dataset_1_sigma)
    test_1_conf_mat, test_1_score_mat = utl.confusion_score_matrix(test_1.y, test_1_pred)
    test_1_acc = utl.calc_accuracy(np.array(test_1.y), test_1_pred)

    # Dataset 2 Data Preparation
    dataset_2_mu = [[3, 6], [5, 4], [6, 6]]
    dataset_2_sigma = [[[1.5, 0.1], [0.1, 0.5]], [[1, -0.2], [-0.2, 2]], [[2, -0.25], [-0.25, 1.5]]]

    dataset_2 = utl.generate_dataset(dataset_2_mu, dataset_2_sigma, dataset_classes, dataset_class_size)

    train_2, test_2 = train_test_split(dataset_2, test_size=0.2)

    # Train Phase Of Dataset 2
    dataset_2_phi, dataset_2_mu, dataset_2_sigma = utl.calc_params(train_2)

    # Test Phase Of Dataset 2
    train_2_pred = utl.bayesian_prediction(train_2, dataset_2_phi, dataset_2_mu, dataset_2_sigma)
    train_2_conf_mat, train_2_score_mat = utl.confusion_score_matrix(train_2.y, train_2_pred)
    train_2_acc = utl.calc_accuracy(np.array(train_2.y), train_2_pred)

    test_2_pred = utl.bayesian_prediction(test_2, dataset_2_phi, dataset_2_mu, dataset_2_sigma)
    test_2_conf_mat, test_2_score_mat = utl.confusion_score_matrix(test_2.y, test_2_pred)
    test_2_acc = utl.calc_accuracy(np.array(test_2.y), test_2_pred)

    # Plots
    # Data Based On Label & Being Correct Or Not
    db_fig, db_axs = plt.subplots(2, 2, figsize=(10.5, 7))
    db_fig.suptitle('Bayesian Classifier With Linear Boundary')
    utl.plot_raw_data(train_1, train_1_pred, 'quadratic', dataset_1_phi, dataset_1_mu, dataset_1_sigma,
                      db_axs[0, 0], "BC-Train1")
    utl.plot_raw_data(test_1, test_1_pred, 'quadratic', dataset_1_phi, dataset_1_mu, dataset_1_sigma,
                      db_axs[1, 0], "BC-Test1")
    utl.plot_raw_data(train_2, train_2_pred, 'quadratic', dataset_2_phi, dataset_2_mu, dataset_2_sigma,
                      db_axs[0, 1], "BC-Train2")
    utl.plot_raw_data(test_2, test_2_pred, 'quadratic', dataset_2_phi, dataset_2_mu, dataset_2_sigma,
                      db_axs[1, 1], "BC-Test2")
    db_fig.tight_layout()

    # PDF & Contour
    dataset_1_x_bound, dataset_1_y_bound = [0, 9.5], [0, 9.5]
    dataset_2_x_bound, dataset_2_y_bound = [0, 9.5], [0, 9.5]
    color_map = ['summer', 'autumn', 'winter']

    pdf_c_fig = plt.figure(figsize=(10, 10))
    pdf_c_fig.suptitle('PDF & Contour Plots')

    pdf_c_axs_pdf1 = pdf_c_fig.add_subplot(2, 2, 1, projection='3d')
    utl.plot_pdf(dataset_1_mu, dataset_1_sigma, pdf_c_axs_pdf1, dataset_1_x_bound, dataset_1_y_bound,
                 color_map, 'Dataset 1 PDF')

    pdf_c_axs_pdf2 = pdf_c_fig.add_subplot(2, 2, 2, projection='3d')
    utl.plot_pdf(dataset_2_mu, dataset_2_sigma, pdf_c_axs_pdf2, dataset_2_x_bound, dataset_2_y_bound,
                 color_map, 'Dataset 2 PDF')
    pdf_c_axs_c1 = pdf_c_fig.add_subplot(2, 2, 3)

    utl.plot_contour(dataset_1.iloc[:, :-1].values, dataset_1_phi, dataset_1_mu, dataset_1_sigma,
                     pdf_c_axs_c1, dataset_1_x_bound, dataset_1_y_bound, color_map, 'quadratic', 'Dataset 1 Contour')
    pdf_c_axs_c2 = pdf_c_fig.add_subplot(2, 2, 4)

    utl.plot_contour(dataset_2.iloc[:, :-1].values, dataset_2_phi, dataset_2_mu, dataset_2_sigma,
                     pdf_c_axs_c2, dataset_2_x_bound, dataset_2_y_bound, color_map, 'quadratic', 'Dataset 2 Contour')

    pdf_c_fig.tight_layout()

    plt.show()

    # Decision Boundary With Scikit-Learn QDA
    dataset_1_train_0v1 = [train_1[train_1.y != 2].iloc[:, :-1].values, train_1[train_1.y != 2].iloc[:, -1].values]
    dataset_2_train_0v1 = [train_2[train_2.y != 2].iloc[:, :-1].values, train_2[train_2.y != 2].iloc[:, -1].values]
    dataset_1_test_0v1 = [test_1[test_1.y != 2].iloc[:, :-1].values, test_1[test_1.y != 2].iloc[:, -1].values]
    dataset_2_test_0v1 = [test_2[test_2.y != 2].iloc[:, :-1].values, test_2[test_2.y != 2].iloc[:, -1].values]

    dataset_1_train_0v2 = [train_1[train_1.y != 1].iloc[:, :-1].values,
                           np.array([1 if y == 2 else 0 for y in train_1[train_1.y != 1].iloc[:, -1].values])]
    dataset_2_train_0v2 = [train_2[train_2.y != 1].iloc[:, :-1].values,
                           np.array([1 if y == 2 else 0 for y in train_2[train_2.y != 1].iloc[:, -1].values])]
    dataset_1_test_0v2 = [test_1[test_1.y != 1].iloc[:, :-1].values,
                          np.array([1 if y == 2 else 0 for y in test_1[test_1.y != 1].iloc[:, -1].values])]
    dataset_2_test_0v2 = [test_2[test_2.y != 1].iloc[:, :-1].values,
                          np.array([1 if y == 2 else 0 for y in test_2[test_2.y != 1].iloc[:, -1].values])]

    dataset_1_train_1v2 = [train_1[train_1.y != 0].iloc[:, :-1].values,
                           np.array([1 if y == 2 else 0 for y in train_1[train_1.y != 0].iloc[:, -1].values])]
    dataset_2_train_1v2 = [train_2[train_2.y != 0].iloc[:, :-1].values,
                           np.array([1 if y == 2 else 0 for y in train_2[train_2.y != 0].iloc[:, -1].values])]
    dataset_1_test_1v2 = [test_1[test_1.y != 0].iloc[:, :-1].values,
                          np.array([1 if y == 2 else 0 for y in test_1[test_1.y != 0].iloc[:, -1].values])]
    dataset_2_test_1v2 = [test_2[test_2.y != 0].iloc[:, :-1].values,
                          np.array([1 if y == 2 else 0 for y in test_2[test_2.y != 0].iloc[:, -1].values])]

    dataset_0v1 = [dataset_1_train_0v1, dataset_2_train_0v1, dataset_1_test_0v1, dataset_2_test_0v1]
    dataset_0v2 = [dataset_1_train_0v2, dataset_2_train_0v2, dataset_1_test_0v2, dataset_2_test_0v2]
    dataset_1v2 = [dataset_1_train_1v2, dataset_2_train_1v2, dataset_1_test_1v2, dataset_2_test_1v2]

    da.plot_decision_boundary(dataset_0v1, 'quadratic', ['Dataset 1 Train 0 VS 1', 'Dataset 2 Train 0 VS 1',
                                                         'Dataset 1 Test 0 VS 1', 'Dataset 2 Test 0 VS 1'])
    da.plot_decision_boundary(dataset_0v2, 'quadratic', ['Dataset 1 Train 0 VS 2', 'Dataset 2 Train 0 VS 2',
                                                         'Dataset 1 Test 0 VS 2', 'Dataset 2 Test 0 VS 2'])
    da.plot_decision_boundary(dataset_1v2, 'quadratic', ['Dataset 1 Train 1 VS 2', 'Dataset 2 Train 1 VS 2',
                                                         'Dataset 1 Test 1 VS 2', 'Dataset 2 Test 1 VS 2'])

    # Results
    print('─' * 50)

    print('Results:')

    print('─' * 50)

    print('Dataset 1 Parameters:\n')
    print(f'Phi:\n{dataset_1_phi}\n\nMu:\n{dataset_1_mu}\n\nSigma:\n{dataset_1_sigma}')

    print('─' * 50)

    print(f'Dataset 1 Train Confusion Matrix:\n{train_1_conf_mat}\n')
    print(f'Dataset 1 Train Score Matrix:\n{train_1_score_mat}\n')
    print(f'Dataset 1 Train Accuracy: {train_1_acc}')

    print('─' * 50)

    print(f'Dataset 1 Test Confusion Matrix:\n{test_1_conf_mat}\n')
    print(f'Dataset 1 Test Score Matrix:\n{test_1_score_mat}\n')
    print(f'Dataset 1 Test Accuracy: {test_1_acc}')

    print('─' * 50)

    print('Dataset 2 Parameters:\n')
    print(f'Phi:\n{dataset_2_phi}\n\nMu:\n{dataset_2_mu}\n\nSigma:\n{dataset_2_sigma}')

    print('─' * 50)

    print(f'Dataset 2 Train Confusion Matrix:\n{train_2_conf_mat}\n')
    print(f'Dataset 2 Train Score Matrix:\n{train_2_score_mat}\n')
    print(f'Dataset 2 Train Accuracy: {train_2_acc}')

    print('─' * 50)

    print(f'Dataset 2 Test Confusion Matrix:\n{test_2_conf_mat}\n')
    print(f'Dataset 2 Test Score Matrix:\n{test_2_score_mat}\n')
    print(f'Dataset 2 Test Accuracy: {test_2_acc}')

    print('─' * 50)
