import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.random_projection import GaussianRandomProjection
from scipy.spatial.distance import cdist
from sklearn import mixture

from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

import base_experiment

phishing_path = ""
if len(sys.argv) > 1:
    phishing_path = sys.argv[1]
    print("Reading data from provided path {}".format(phishing_path))
else:
    phishing_path = 'data/breast-cancer-wisconsin.data'

names = ['Sample code number', 'Clump Thickness',
         'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
         'Single Epithelial Cell Size',
         'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
         'Mitoses', 'Class']

features = ['Clump Thickness',
            'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
            'Single Epithelial Cell Size',
            'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
            'Mitoses']
target = 'Class'

plot_name = "Breast Cancer Detection"

data = pd.read_csv(phishing_path, names=names, header=None)

data = data.apply(pd.to_numeric, args=('coerce',))

# print(data.dtypes)
data = data.dropna()

# print(data)
features_data = data[features]

top_2_features = base_experiment.identify_top_2_features(features_data)

x_train = preprocessing.scale(features_data)

base_experiment.plot_points(plot_name, x_train, top_2_features)

y_train = data[target]

base_experiment.plot_score_em(range(2, 20), x_train, plot_name)
base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), x_train, plot_name)
base_experiment.plot_silhoutte_score_kmeans(range(2, 20), x_train, plot_name)
base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), x_train, plot_name)
#
kmeans_best_k = 5
em_best_k = 12

ica_best_components = 3
pca_best_components = 3

clfr = KMeans(n_clusters=kmeans_best_k)
clfr.fit(x_train)

base_experiment.describe_what_you_see(clfr.predict(x_train), y_train, plot_name,
                                      kmeans_best_k, "K Means (k={})".format(kmeans_best_k))
base_experiment.plot_points(plot_name, x_train, top_2_features, clfr.predict(x_train), clfr.cluster_centers_,
                            kmeans_best_k, "K Means k={}".format(kmeans_best_k))

clfr = mixture.GaussianMixture(n_components=em_best_k, covariance_type='full')
clfr.fit(x_train)
base_experiment.describe_what_you_see(clfr.predict(x_train), y_train, plot_name,
                                      kmeans_best_k, "EM (k={})".format(em_best_k))
base_experiment.plot_points(plot_name, x_train, top_2_features, clfr.predict(x_train), clfr.means_,
                            kmeans_best_k, "EM k={}".format(em_best_k))

base_experiment.run_pca_and_plot(x_train, plot_name)

pca = PCA(n_components=pca_best_components)
pca_x_train = pca.fit_transform(x_train)
base_experiment.plot_points_3d("{}-{}".format(plot_name, "PCA"), pca_x_train)

base_experiment.run_ica_and_plot(x_train, plot_name, len(features))

ica = FastICA(n_components=ica_best_components)
ica_x_train = ica.fit_transform(x_train)
base_experiment.plot_points_3d("{}-{}".format(plot_name, "ICA"), ica_x_train)


base_experiment.selecting_features_with_low_variance(plot_name, features_data, features, 0.8)

# plot_points("{}:KMeans".format(plot_name), data[features].values, top_2_features)
# print(np.shape(y_train))
# print(np.shape(clfr.predict(x_train)))
#
