# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 01:07:25 2024

@author: Owner

This script is designed to demonstrate the application of Kernel PCA on a Swiss roll dataset for
dimensionality reduction, followed by logistic regression optimization using GridSearchCV.
The objective is to find the best kernel type and gamma value for Kernel PCA to enhance classification accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def generate_swiss_roll(n_samples=1000):
    """
    Generates a Swiss roll dataset with a specified number of samples.

    Parameters:
    n_samples (int): The number of samples to generate.

    Returns:
    X (numpy.ndarray): The generated data (3D coordinates for each sample).
    t (numpy.ndarray): The generated labels (continuous variable associated with each sample).
    """
    X, t = make_swiss_roll(n_samples)
    return X, t

def plot_swiss_roll(X, t):
    """
    Plots the Swiss roll in a 3D space using matplotlib.

    Parameters:
    X (numpy.ndarray): The data points to plot.
    t (numpy.ndarray): The color coding for each point based on labels.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
    ax.set_title('Swiss Roll')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def apply_kernel_pca(X, kernel_type='linear', n_components=2):
    """
    Applies Kernel PCA to the data with a specified kernel type.

    Parameters:
    X (numpy.ndarray): The data on which to perform PCA.
    kernel_type (str): The type of kernel to use ('linear', 'rbf', 'sigmoid').

    Returns:
    X_kpca (numpy.ndarray): The transformed data after applying Kernel PCA.
    """
    kpca = KernelPCA(n_components=n_components, kernel=kernel_type)
    X_kpca = kpca.fit_transform(X)
    return X_kpca

def plot_kpca_results(X_transformed, t, title):
    """
    Plots the results of Kernel PCA in 2D space.

    Parameters:
    X_transformed (numpy.ndarray): The data after Kernel PCA transformation.
    t (numpy.ndarray): The color coding for each point based on labels.
    title (str): The title for the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=t, cmap=plt.cm.hot)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()

def optimize_kpca_logistic_regression(X, t):
    """
    Uses GridSearchCV to optimize Kernel PCA and logistic regression in a pipeline.

    Parameters:
    X (numpy.ndarray): The data to fit.
    t (numpy.ndarray): The labels for the data.

    Returns:
    grid_search (GridSearchCV object): The GridSearchCV object after fitting.
    """
    pipeline = Pipeline([
        ('kpca', KernelPCA(n_components=2)),
        ('log_reg', LogisticRegression())
    ])

    param_grid = {
        'kpca__kernel': ['linear', 'rbf', 'sigmoid'],
        'kpca__gamma': np.linspace(0.03, 0.05, 10)
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3)
    grid_search.fit(X, t > np.median(t))
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    return grid_search

def main():
    """
    Main function to orchestrate the data generation, PCA application, visualization, and optimization tasks.
    """
    X, t = generate_swiss_roll()
    plot_swiss_roll(X, t)
    
    # Apply and plot Kernel PCA with different kernels
    for kernel in ['linear', 'rbf', 'sigmoid']:
        X_kpca = apply_kernel_pca(X, kernel)
        plot_kpca_results(X_kpca, t, f'{kernel.capitalize()} Kernel PCA')
    
    # Optimize Kernel PCA and logistic regression
    grid_search = optimize_kpca_logistic_regression(X, t)
    
    # Plot results from GridSearchCV
    X_kpca_optimized = grid_search.best_estimator_['kpca'].transform(X)
    plot_kpca_results(X_kpca_optimized, t, 'Optimized Kernel PCA')

if __name__ == "__main__":
    main()
