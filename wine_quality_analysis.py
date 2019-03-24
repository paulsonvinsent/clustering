import sys

import pandas as pd
from sklearn import mixture
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier

from helper import plot_learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
import numpy as np

import base_experiment

wine_quality_path = ""
if len(sys.argv) > 1:
    wine_quality_path = sys.argv[1]
    print("Reading data from provided path {}".format(wine_quality_path))
else:
    wine_quality_path = 'data/winequality-red.csv'

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
target = 'quality'

plot_name = "Wine Quality"

data = pd.read_csv(wine_quality_path)

features_data = data[features]

top_2_features = base_experiment.identify_top_2_features(features_data)

x_train = preprocessing.scale(features_data)

base_experiment.plot_points(plot_name, x_train, top_2_features)

y_train = data[target]

standard_scaler = StandardScaler()

min_max_scaler = MinMaxScaler(feature_range=(0, 1))
min_max_scaler.fit(features_data)


def simple_clustering(plot_name, X, kmeans_k, em_k, top_2_features_given=None):
    if top_2_features_given is None:
        top_2_features_given = top_2_features
    else:
        top_2_features_given = base_experiment.identify_top_2_features(X)

    base_experiment.plot_score_em(range(2, 20), X, plot_name)
    base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), X, plot_name)
    base_experiment.plot_silhoutte_score_kmeans(range(2, 20), X, plot_name)
    base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), X, plot_name)
    clfr = KMeans(n_clusters=kmeans_k)
    clfr.fit(X)
    kmeans_clusters = clfr.predict(X)
    base_experiment.describe_what_you_see(kmeans_clusters, y_train, plot_name,
                                          kmeans_k, "K Means (k={})".format(kmeans_k))
    base_experiment.plot_points(plot_name, X, top_2_features_given, clfr.predict(X), clfr.cluster_centers_,
                                kmeans_k, "K Means k={}".format(kmeans_k))
    clfr = mixture.GaussianMixture(n_components=em_k, covariance_type='full')
    clfr.fit(X)
    em_clusters = clfr.predict(X)
    base_experiment.describe_what_you_see(em_clusters, y_train, plot_name,
                                          kmeans_k, "EM (k={})".format(em_k))
    base_experiment.plot_points(plot_name, X, top_2_features_given, clfr.predict(X), clfr.means_,
                                kmeans_k, "EM k={}".format(em_k))
    kmeans_clusters = kmeans_clusters.reshape((1599, 1))
    em_clusters = em_clusters.reshape((1599, 1))
    concatenate = np.concatenate((kmeans_clusters, em_clusters), axis=1)
    return concatenate


def explore_dimensionality_reduction():
    base_experiment.run_pca_and_plot(x_train, plot_name, y_train)
    base_experiment.run_ica_and_plot(x_train, plot_name, len(features), y_train)
    base_experiment.plot_features_and_variance(plot_name, min_max_scaler.transform(features_data), features)


def dimensionality_reduction():
    ica_best_components = 5
    pca_best_components = 6
    rp_chosen_components = 3
    variance_threshold = 0.02
    pca = PCA(n_components=pca_best_components)
    pca_x_train = pca.fit_transform(x_train)
    base_experiment.plot_eigen_values("{}-{}".format(plot_name, "PCA"), pca.explained_variance_)
    base_experiment.plot_points_3d("{}-{}".format(plot_name, "PCA"), pca_x_train)
    ica = FastICA(n_components=ica_best_components)
    ica_x_train = ica.fit_transform(x_train)
    base_experiment.plot_points_3d("{}-{}".format(plot_name, "ICA"), ica_x_train)
    rp = GaussianRandomProjection(n_components=rp_chosen_components)
    rp_x_train = rp.fit_transform(x_train)
    base_experiment.plot_points_3d("{}-{}".format(plot_name, "Random Projection"), rp_x_train)
    variance_x_train = VarianceThreshold(threshold=variance_threshold).fit_transform(
        min_max_scaler.transform(features_data))
    variance_x_train = preprocessing.scale(variance_x_train)
    find_best_k_for_reduced_features(ica_x_train, pca_x_train, rp_x_train, variance_x_train)
    clustering_after_reduction(pca_x_train, ica_x_train, rp_x_train, variance_x_train)
    run_ann_with_only_dimensionality_reduction(pca_x_train, ica_x_train, rp_x_train, variance_x_train)


def run_ann_with_only_dimensionality_reduction(pca_x_train, ica_x_train, rp_x_train, variance_x_train):
    classifier = MLPClassifier(hidden_layer_sizes=(25), activation='logistic',
                               max_iter=5000, solver='adam')
    cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    ann_data = standard_scaler.fit_transform(data[features])
    plot_learning_curve("{}- DR - ANN Learning Curve".format(plot_name), classifier,
                        "{}- DR - ANN Learning Curve Simple".format(plot_name), ann_data,
                        data[target], ylim=(0, 1), cv=cv,
                        n_jobs=4)
    plot_learning_curve("{}-PCA DR - ANN Learning Curve".format(plot_name), classifier,
                        "{}- PCA DR - ANN Learning Curve".format(plot_name), standard_scaler.fit_transform(pca_x_train),
                        data[target], ylim=(0, 1), cv=cv,
                        n_jobs=4)
    plot_learning_curve("{}-ICA DR - ANN Learning Curve".format(plot_name), classifier,
                        "{}- ICA DR - ANN Learning Curve".format(plot_name), standard_scaler.fit_transform(ica_x_train),
                        data[target], ylim=(0, 1), cv=cv,
                        n_jobs=4)
    plot_learning_curve("{}-RP DR - ANN Learning Curve".format(plot_name), classifier,
                        "{}- RP DR - ANN Learning Curve".format(plot_name), standard_scaler.fit_transform(rp_x_train),
                        data[target], ylim=(0, 1), cv=cv,
                        n_jobs=4)
    plot_learning_curve("{}-Variance filter DR - ANN Learning Curve".format(plot_name), classifier,
                        "{}- Variance filter DR - ANN Learning Curve".format(plot_name),
                        standard_scaler.fit_transform(variance_x_train),
                        data[target], ylim=(0, 1), cv=cv,
                        n_jobs=4)


def find_best_k_for_reduced_features(ica_x_train, pca_x_train, rp_x_train, variance_x_train):
    base_experiment.plot_points_3d("{}-{}".format(plot_name, "Variance Threshold"), rp_x_train)
    base_experiment.plot_score_em(range(2, 20), pca_x_train, '{}-PCA-'.format(plot_name))
    base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), pca_x_train, '{}-PCA'.format(plot_name))
    base_experiment.plot_silhoutte_score_kmeans(range(2, 20), pca_x_train, '{}-PCA'.format(plot_name))
    base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), pca_x_train, '{}-PCA'.format(plot_name))
    base_experiment.plot_score_em(range(2, 20), ica_x_train, '{}-ICA-'.format(plot_name))
    base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), ica_x_train, '{}-ICA'.format(plot_name))
    base_experiment.plot_silhoutte_score_kmeans(range(2, 20), ica_x_train, '{}-ICA'.format(plot_name))
    base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), ica_x_train, '{}-ICA'.format(plot_name))
    base_experiment.plot_score_em(range(2, 20), rp_x_train, '{}-RP-'.format(plot_name))
    base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), rp_x_train, '{}-RP'.format(plot_name))
    base_experiment.plot_silhoutte_score_kmeans(range(2, 20), rp_x_train, '{}-RP'.format(plot_name))
    base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), rp_x_train, '{}-RP'.format(plot_name))
    base_experiment.plot_score_em(range(2, 20), variance_x_train, '{}-Variance Threshold-'.format(plot_name))
    base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), variance_x_train,
                                                   '{}-Variance Threshold'.format(plot_name))
    base_experiment.plot_silhoutte_score_kmeans(range(2, 20), variance_x_train,
                                                '{}-Variance Threshold'.format(plot_name))
    base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), variance_x_train,
                                                   '{}-Variance Threshold'.format(plot_name))


def clustering_after_reduction(pca_x_train, ica_x_train, rp_x_train, variance_x_train):
    ica_kmeans_k = 6
    ica_em_k = 5
    pca_em_k = 5
    pca_kmeans_k = 7
    rp_em_k = 6
    rp_kmeans_k = 7
    variance_filter_em_k = 13
    variance_filter_kmeans_k = 6
    cluster_params1 = simple_clustering("{}- ICA".format(plot_name), ica_x_train, ica_kmeans_k, ica_em_k,
                                        base_experiment.identify_top_2_features(ica_x_train))
    cluster_params2 = simple_clustering("{}- PCA".format(plot_name), pca_x_train, pca_kmeans_k, pca_em_k,
                                        base_experiment.identify_top_2_features(pca_x_train))
    cluster_params3 = simple_clustering("{}- Random Projection".format(plot_name), rp_x_train, rp_kmeans_k, rp_em_k,
                                        base_experiment.identify_top_2_features(rp_x_train))
    cluster_params4 = simple_clustering("{}- Variance Filtering".format(plot_name), variance_x_train,
                                        variance_filter_kmeans_k,
                                        variance_filter_em_k, base_experiment.identify_top_2_features(variance_x_train))
    final_cluster_array = np.concatenate((cluster_params1, cluster_params2, cluster_params3, cluster_params4), axis=1)
    cluster_features = pd.DataFrame(data=final_cluster_array)
    classifier = MLPClassifier(hidden_layer_sizes=(25), activation='logistic',
                               max_iter=5000, solver='adam')
    cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    plot_learning_curve("{}-Cluster Features - ANN Learning Curve".format(plot_name), classifier,
                        "{}- Cluster Features - ANN Learning Curve".format(plot_name),
                        standard_scaler.fit_transform(cluster_features),
                        data[target], ylim=(0, 1), cv=cv,
                        n_jobs=4)


kmeans_best_k = 8
em_best_k = 7

# simple_clustering(plot_name, x_train, kmeans_best_k, em_best_k, top_2_features)
# explore_dimensionality_reduction()
dimensionality_reduction()
