import sys

import pandas as pd
from sklearn import mixture
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

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
    base_experiment.describe_what_you_see(clfr.predict(X), y_train, plot_name,
                                          kmeans_k, "K Means (k={})".format(kmeans_k))
    base_experiment.plot_points(plot_name, X, top_2_features_given, clfr.predict(X), clfr.cluster_centers_,
                                kmeans_k, "K Means k={}".format(kmeans_k))
    clfr = mixture.GaussianMixture(n_components=em_k, covariance_type='full')
    clfr.fit(X)
    base_experiment.describe_what_you_see(clfr.predict(X), y_train, plot_name,
                                          kmeans_k, "EM (k={})".format(em_k))
    base_experiment.plot_points(plot_name, X, top_2_features_given, clfr.predict(X), clfr.means_,
                                kmeans_k, "EM k={}".format(em_k))
    base_experiment.run_pca_and_plot(X, plot_name)


def dimensionality_reduction():
    ica_best_components = 3
    pca_best_components = 3
    rp_chosen_components = 3
    variance_threshold = 9
    pca = PCA(n_components=pca_best_components)
    pca_x_train = pca.fit_transform(x_train)
    base_experiment.plot_points_3d("{}-{}".format(plot_name, "PCA"), pca_x_train)
    base_experiment.run_ica_and_plot(x_train, plot_name, len(features))
    ica = FastICA(n_components=ica_best_components)
    ica_x_train = ica.fit_transform(x_train)
    base_experiment.plot_points_3d("{}-{}".format(plot_name, "ICA"), ica_x_train)
    rp = GaussianRandomProjection(n_components=rp_chosen_components)
    rp_x_train = rp.fit_transform(x_train)
    base_experiment.plot_points_3d("{}-{}".format(plot_name, "Random Projection"), rp_x_train)
    variance_x_train = base_experiment.selecting_features_with_low_variance(plot_name, features_data, features,
                                                                            variance_threshold)
    # find_best_k_for_reduced_features(ica_x_train, pca_x_train, rp_x_train, variance_x_train)

    clustering_after_reduction(pca_x_train, ica_x_train, rp_x_train, variance_x_train)


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
    ica_kmeans_k = 5
    ica_em_k = 8
    pca_em_k = 4
    pca_kmeans_k = 5
    rp_em_k = 12
    rp_kmeans_k = 5
    variance_filter_em_k = 9
    variance_filter_kmeans_k = 10
    simple_clustering("{}- ICA".format(plot_name), ica_x_train, ica_kmeans_k, ica_em_k,
                      base_experiment.identify_top_2_features(ica_x_train))
    simple_clustering("{}- PCA".format(plot_name), pca_x_train, pca_kmeans_k, pca_em_k,
                      base_experiment.identify_top_2_features(pca_x_train))
    simple_clustering("{}- Random Projection".format(plot_name), rp_x_train, rp_kmeans_k, rp_em_k,
                      base_experiment.identify_top_2_features(rp_x_train))
    simple_clustering("{}- Variance Filtering".format(plot_name), variance_x_train, variance_filter_kmeans_k,
                      variance_filter_em_k, base_experiment.identify_top_2_features(variance_x_train))


kmeans_best_k = 8
em_best_k = 7

simple_clustering(plot_name, x_train, kmeans_best_k, em_best_k, top_2_features)
dimensionality_reduction()
