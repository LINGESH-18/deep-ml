import numpy as np


def instance_normalization(X, gamma, beta, eps=1e-5):

    # Step 1: Get tensor dimensions
    B, C, H, W = X.shape

    # Step 2: Calculate spatial mean per instance and channel
    mean = np.mean(
        X,
        axis=(2, 3),
        keepdims=True
    )

    # Step 3: Calculate spatial variance per instance and channel
    variance = np.var(
        X,
        axis=(2, 3),
        keepdims=True
    )

    # Step 4: Normalize
    X_normalized = (
        X - mean
    ) / np.sqrt(variance + eps)

    # Step 5: Reshape gamma and beta for broadcasting
    gamma = gamma.reshape(1, C, 1, 1)
    beta = beta.reshape(1, C, 1, 1)

    # Step 6: Apply learnable scale and shift
    out = gamma * X_normalized + beta

    return out