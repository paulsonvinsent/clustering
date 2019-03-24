import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import cdist
from sklearn import mixture

from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
from sklearn.random_projection import GaussianRandomProjection

random.seed(100)
np.random.seed(100)


def plot_elbow_method_graph_kmeans(K, X, name):
    plt.plot()
    # k means determine k
    distortions = []
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('{} KMeans : The Elbow Method showing the optimal k'.format(name))
    plt.savefig('plots/{}-elbow-kmeans.png'.format(name))


def plot_score_em(K, X, name):
    # k means determine k
    silhouette_scores = []
    for k in K:
        clfr = mixture.GaussianMixture(n_components=k, covariance_type='full')
        clfr.fit(X)
        silhouette_scores.append(silhouette_score(X, clfr.predict(X)))

    # Plot the elbow
    plt.plot()
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Scores')
    plt.title('{} EM : Silhouette Method showing the optimal k'.format(name))
    plt.savefig('plots/{}-silhouette-em.png'.format(name))


def plot_silhoutte_score_kmeans(K, X, name):
    plt.plot()
    # k means determine k
    silhouette_scores = []
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        silhouette_scores.append(silhouette_score(X, kmeanModel.predict(X)))

    # Plot the elbow
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Scores')
    plt.title('{} KMeans : Silhouette Method showing the optimal k'.format(name))
    plt.savefig('plots/{}-silhouette-kmeans.png'.format(name))


def describe_what_you_see(cluster_labels,
                          classification_labels,
                          name,
                          number_of_clusters,
                          algorithm_name):
    unique_cluster_labels = np.unique(cluster_labels)
    for cluster in unique_cluster_labels:
        plt.plot()
        classification_labels_in_this_cluster = classification_labels[cluster_labels == cluster]

        # print(classification_labels_in_this_cluster)
        unique, counts = np.unique(classification_labels_in_this_cluster, return_counts=True)
        y_pos = np.arange(len(unique))
        y_labels = unique.tolist()
        plt.bar(y_pos, counts, align='center', alpha=0.5)
        plt.xticks(y_pos, y_labels)
        plt.ylabel('Counts')
        plt.title('{} : Simple clustering results ,Cluster {} '.format(name, cluster))
        plt.savefig('plots/{}-{}-cluster-{}-labels.png'.format(name, algorithm_name, cluster))
    plt.figure()
    plt.hist(cluster_labels, bins=np.arange(0, number_of_clusters + 1) - 0.5, rwidth=0.5, zorder=2)
    plt.xticks(np.arange(0, number_of_clusters))
    plt.xlabel('Cluster label')
    plt.ylabel('Number of samples')
    plt.title('Dataset: {}'.format(name))
    plt.grid()
    plt.savefig('plots/{}-{}-cluster-counts.png'.format(name, algorithm_name))


def run_pca_and_plot(X, name):
    pca = PCA()
    pca.fit(X)
    plt.figure()
    plt.plot(np.arange(1, pca.explained_variance_ratio_.size + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.xticks(np.arange(1, pca.explained_variance_ratio_.size + 1))
    plt.xlabel('Components')
    plt.ylabel('Variance')
    plt.title('{} : PCA variance'.format(name))
    plt.grid()
    fig = plt.figure()
    fig.savefig('plots/{}-pca-variance.png'.format(name), dpi=fig.dpi)
    # plot recunstruction error
    reconstruction_error = []
    for n_components in np.arange(1, pca.explained_variance_ratio_.size + 1):
        pca = PCA(n_components=n_components)
        reconstruction_error.append(np.sum(np.square(X - pca.inverse_transform(pca.fit_transform(X)))) / X.size)
    plt.figure()
    plt.plot(np.arange(1, pca.explained_variance_ratio_.size + 1), reconstruction_error)
    plt.xticks(np.arange(1, pca.explained_variance_ratio_.size + 1))
    plt.xlabel('Components')
    plt.ylabel('Reconstruction Error')
    plt.title('{} : PCA Reconstruction Error'.format(name))
    plt.grid()
    plt.savefig('plots/{}-pca-reconstruction.png'.format(name))


def run_ica_and_plot(X, name, number_of_features):
    # plot recunstruction error
    reconstruction_error = []
    for n_components in np.arange(1, number_of_features + 1):
        ica = FastICA(n_components=n_components)
        reconstruction_error.append(np.sum(np.square(X - ica.inverse_transform(ica.fit_transform(X)))) / X.size)
    plt.figure()
    plt.plot(np.arange(1, number_of_features + 1), reconstruction_error)
    plt.xticks(np.arange(1, number_of_features + 1))
    plt.xlabel('Components')
    plt.ylabel('Reconstruction Error')
    plt.title('{} : ICA Reconstruction Error'.format(name))
    plt.grid()
    plt.savefig('plots/{}-ica-reconstruction.png'.format(name))


def run_random_projection_and_plot(X, name, number_of_componenets):
    # plot recunstruction error
    reconstruction_error = []
    for n_components in np.arange(1, number_of_componenets + 1):
        ica = GaussianRandomProjection(n_components=n_components)
        reconstruction_error.append(np.sum(np.square(X - ica.inverse_transform(ica.fit_transform(X)))) / X.size)
    plt.figure()
    plt.plot(np.arange(1, number_of_componenets + 1), reconstruction_error)
    plt.xticks(np.arange(1, number_of_componenets + 1))
    plt.xlabel('Components')
    plt.ylabel('Reconstruction Error')
    plt.title('{} : Random projection Reconstruction Error'.format(name))
    plt.grid()
    plt.savefig('plots/{}-random-projection-reconstruction.png'.format(name))


def selecting_features_with_low_variance(name, X, features, variance_threshold):
    variances = np.var(X, axis=0)
    plt.figure()
    plt.bar(np.arange(len(features)), variances.tolist(), align='center', alpha=0.5)
    plt.xticks(np.arange(len(features)), np.arange(len(features)))
    plt.xlabel('Features')
    plt.ylabel('Variance')
    plt.title("{} : Features and Variances".format(name))
    plt.grid()
    plt.savefig('plots/{}-variance-based-filtering.png'.format(name))
    sel = VarianceThreshold(threshold=variance_threshold)
    return sel.fit_transform(X)


def identify_top_2_features(X):
    variances = np.var(X, axis=0)
    tolist = variances.tolist()
    return sorted(range(len(tolist)), key=lambda x: tolist[x])[-2:]


def plot_points(name, X, top_2_features, classification=None, centers=None, k=None, algorithm_name=None):
    if classification is not None:
        cmap = cm.get_cmap("Spectral")
        colors = cmap(classification.astype(float) / k)
        plt.figure()
        plt.scatter(X[:, top_2_features[0]], X[:, top_2_features[1]], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
        plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
        plt.title("{}:{} The visualization of the clustered data.".format(name, algorithm_name))
        plt.xlabel("Feature space for the 1st feature")
        plt.ylabel("Feature space for the 2nd feature")
        plt.savefig('plots/{}-{}-clusters.png'.format(name, algorithm_name))
    else:
        plt.figure()
        plt.scatter(X[:, top_2_features[0]], X[:, top_2_features[1]], marker='.', s=30, lw=0, alpha=0.7, edgecolor='k')
        plt.title("{}: The visualization of points".format(name))
        plt.xlabel("Feature space for the 1st feature")
        plt.ylabel("Feature space for the 2nd feature")
        plt.savefig('plots/{}-points.png'.format(name))
