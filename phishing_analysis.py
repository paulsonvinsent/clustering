import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.cm as cm

random.seed(100)
np.random.seed(100)

best_k_for_kmeans = 4

best_k_for_em = 3


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
    plt.show()


def plot_score_em(K, X, name):
    # k means determine k
    silhouette_scores = []
    bic_scores = []
    for k in K:
        clfr = mixture.GaussianMixture(n_components=k, covariance_type='full')
        clfr.fit(X)
        bic_scores.append(clfr.bic(X))
        silhouette_scores.append(silhouette_score(X, clfr.predict(X)))

    # Plot the elbow
    plt.plot()
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Scores')
    plt.title('{} EM : Silhouette Method showing the optimal k'.format(name))
    plt.show()
    # Plot  bic
    plt.plot()
    plt.plot(K, bic_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('BIC values')
    plt.title('{} EM : BIC Method showing the optimal k'.format(name))
    plt.show()


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
    plt.show()


def describe_what_you_see(cluster_labels,
                          classification_labels, name,
                          X, features, number_of_clusters):
    # print(classification_labels)
    unique_cluster_labels = np.unique(cluster_labels)
    print(unique_cluster_labels)
    for cluster in unique_cluster_labels:
        plt.plot()
        classification_labels_in_this_cluster = classification_labels[cluster_labels == cluster]
        print("what i wannt")
        print(np.shape(classification_labels_in_this_cluster))

        # print(classification_labels_in_this_cluster)
        unique, counts = np.unique(classification_labels_in_this_cluster, return_counts=True)
        print(np.shape(unique))
        print(np.shape(counts))
        print(len(unique))
        y_pos = np.arange(len(unique))
        y_labels = unique.tolist()
        # y_labels = ['0', '1']
        plt.bar(y_pos, counts, align='center', alpha=0.5)
        plt.xticks(y_pos, y_labels)
        print(y_labels)
        plt.ylabel('Counts')
        plt.title('{} : Simple clustering results ,Cluster {} '.format(name, cluster))
        plt.show()
    plt.figure()
    plt.hist(cluster_labels, bins=np.arange(0, number_of_clusters + 1) - 0.5, rwidth=0.5, zorder=2)
    plt.xticks(np.arange(0, number_of_clusters))
    plt.xlabel('Cluster label')
    plt.ylabel('Number of samples')
    plt.title('Dataset: {}'.format(name))
    plt.grid()
    plt.show()


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
    plt.show()
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
    plt.show()


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
    plt.show()


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
    plt.title('{} : ICA Reconstruction Error'.format(name))
    plt.grid()
    plt.show()


def identify_top_2_features(X):
    variances = np.var(X, axis=0)
    tolist = variances.tolist()
    return sorted(range(len(tolist)), key=lambda x: tolist[x])[-2:]


def plot_points(name, X, top_2_features, classification=None, centers=None, k=None):
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
        plt.title("{}: The visualization of the clustered data.".format(name))
        plt.xlabel("Feature space for the 1st feature")
        plt.ylabel("Feature space for the 2nd feature")
        plt.show()
    else:
        plt.figure()
        plt.scatter(X[:, top_2_features[0]], X[:, top_2_features[1]], marker='.', s=30, lw=0, alpha=0.7, edgecolor='k')
        plt.title("{}: The visualization of points".format(name))
        plt.xlabel("Feature space for the 1st feature")
        plt.ylabel("Feature space for the 2nd feature")
        plt.show()


phishing_path = ""
if len(sys.argv) > 1:
    phishing_path = sys.argv[1]
    print("Reading data from provided path {}".format(phishing_path))
else:
    phishing_path = '/Users/pvincent/Desktop/whole desktop/ml-project1/supervised-learning/phishing/phishing.csv'

names = ['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 'double_slash_redirecting',
         'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon', 'port',
         'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email',
         'Abnormal_URL', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
         'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report',
         'Result']

features = ['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 'double_slash_redirecting',
            'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon', 'port',
            'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email',
            'Abnormal_URL', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
            'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report']
target = 'Result'

data = pd.read_csv(phishing_path, names=names, header=None)

top_2_features = identify_top_2_features(data[features])

x_train = preprocessing.scale(data[features])
y_train = data[target]

plot_name = "Phishing Detection"

# plot_score_em(range(2, 20), x_train, plot_name)

# run_pca_and_plot(x_train, plot_name)
# print(np.shape(np.var(data[features], axis=0)))
# print(np.var(data[features], axis=0))
# print((np.var(data[features], axis=0)).argsort()[-2:])

# run_random_projection_and_plot(x_train,plot_name,len(features))

# plot_silhoutte_score_kmeans(range(2, 10), x_train, plot_name)
clfr = KMeans(n_clusters=best_k_for_kmeans)
clfr.fit(x_train)

plot_points("{}:KMeans".format(plot_name), data[features].values, top_2_features, clfr.predict(x_train), clfr.cluster_centers_,
            best_k_for_kmeans)
#



# plot_points("{}:KMeans".format(plot_name), data[features].values, top_2_features)
# print(np.shape(y_train))
# print(np.shape(clfr.predict(x_train)))
#
# describe_what_you_see(clfr.predict(x_train), y_train, '{} - KMeans algorithm'.format(plot_name), x_train, features,
#                       best_k_for_kmeans)
