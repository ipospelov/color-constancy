import numpy as np

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[int(labels[label_idx])]
            label_idx += 1
    return image


# Calculate the covariance matrix for the dataset X
def calculate_covariance_matrix(X, Y=None):
    if not Y:
        Y = X
    X_mean = np.ones(np.shape(X)) * X.mean(0)
    Y_mean = np.ones(np.shape(Y)) * Y.mean(0)
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * (X - X_mean).T.dot(Y - Y_mean)

    return np.array(covariance_matrix, dtype=float)

def calculate_correlation_matrix(X, Y=None):
    if not Y:
        Y = X
    covariance = calculate_covariance_matrix(X, Y)
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_Y = np.expand_dims(calculate_std_dev(Y), 1)
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_Y.T))

    return np.array(correlation_matrix, dtype=float)

# Calculate the standard deviations of the features in dataset X
def calculate_std_dev(X):
    std_dev = np.sqrt(calculate_variance(X))

    return std_dev

# Return the variance of the features in dataset X
def calculate_variance(X):
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))

    return variance