# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 00:40:28 2024

@author: Owner

This script processes the MNIST dataset to perform image recognition tasks using
principal component analysis (PCA). It includes loading data, visualizing digits,
conducting PCA for dimension reduction, and comparing original and compressed images.
"""

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import matplotlib.pyplot as plt

def load_mnist():
    """
    Load the MNIST dataset from OpenML, scale pixel values to [0, 1], and convert to a NumPy array.
    
    Returns:
    X (numpy.ndarray): Image data array with shape (n_samples, 784).
    y (numpy.ndarray): Labels array with shape (n_samples,).
    """
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist['data'], mnist['target']
    X = np.array(X / 255.0)  # Normalize the pixel values to 0-1 range
    return X, y

def plot_digits(X, y):
    """
    Plot the first 10 digits from the MNIST dataset.
    
    Parameters:
    X (numpy.ndarray): Image data array.
    y (numpy.ndarray): Labels array.
    """
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        digit_image = X[i].reshape(28, 28)
        plt.imshow(digit_image, cmap='binary')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.show()

def pca_explained_variance(X):
    """
    Perform PCA on the dataset and retrieve the explained variance ratio for the first two components.
    
    Parameters:
    X (numpy.ndarray): Image data array.
    
    Returns:
    Tuple containing:
        - Explained variance ratio (numpy.ndarray).
        - PCA model (sklearn.decomposition.PCA).
    """
    pca = PCA(n_components=2)
    pca.fit(X)
    return pca.explained_variance_ratio_, pca

def plot_pca_projections(X, y):
    """
    Plot the projection of the MNIST dataset onto the first two principal components.
    
    Parameters:
    X (numpy.ndarray): Image data array.
    y (numpy.ndarray): Labels array.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar()
    plt.show()

def reduce_dimensionality(X):
    """
    Reduce the dimensionality of the dataset using Incremental PCA to 154 dimensions.
    
    Parameters:
    X (numpy.ndarray): Image data array.
    
    Returns:
    Tuple containing:
        - Reduced dimension data (numpy.ndarray).
        - Incremental PCA model (sklearn.decomposition.IncrementalPCA).
    """
    n_batches = 100
    inc_pca = IncrementalPCA(n_components=154)
    for X_batch in np.array_split(X, n_batches):
        inc_pca.partial_fit(X_batch)
    X_reduced = inc_pca.transform(X)
    return X_reduced, inc_pca

def plot_compressed_digits(X, X_reduced, inc_pca):
    """
    Display original and compressed images to compare the effects of dimensionality reduction.
    
    Parameters:
    X (numpy.ndarray): Original image data array.
    X_reduced (numpy.ndarray): Reduced dimension data array.
    inc_pca (sklearn.decomposition.IncrementalPCA): Incremental PCA model.
    """
    X_recovered = inc_pca.inverse_transform(X_reduced)
    fig, axes = plt.subplots(2, 10, figsize=(10, 4), subplot_kw={'xticks':[], 'yticks':[]})
    for i in range(10):
        axes[0, i].imshow(X[i].reshape(28, 28), cmap='binary')
        axes[1, i].imshow(X_recovered[i].reshape(28, 28), cmap='binary')
    plt.show()

def main():
    """
    Main function to orchestrate the data loading, processing, and visualization tasks.
    """
    X, y = load_mnist()
    plot_digits(X, y)
    explained_variance_ratio, pca = pca_explained_variance(X)
    print(f"Explained Variance Ratio: {explained_variance_ratio}")
    plot_pca_projections(X, y)
    X_reduced, inc_pca = reduce_dimensionality(X)
    plot_compressed_digits(X, X_reduced, inc_pca)

if __name__ == "__main__":
    main()
