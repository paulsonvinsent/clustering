import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.random_projection import GaussianRandomProjection

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

data = pd.read_csv(phishing_path, names=names, header=None)

data = data.apply(pd.to_numeric, args=('coerce',))


print(data.dtypes)
data = data.dropna()





print(data)
features_data = data[features]

# top_2_features = base_experiment.identify_top_2_features(features)

x_train = preprocessing.scale(features_data)
y_train = data[target]

plot_name = "Phishing Detection"

base_experiment.plot_score_em(range(2, 20), x_train, plot_name)
base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), x_train, plot_name)
base_experiment.plot_silhoutte_score_kmeans(range(2, 20), x_train, plot_name)
base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), x_train, plot_name)
#
# kmeans_best_k = 3
# em_best_k = 3
#
# clfr = KMeans(n_clusters=kmeans_best_k)
# clfr.fit(x_train)
#
# base_experiment.describe_what_you_see(clfr.predict(x_train), y_train, plot_name,
#                                       kmeans_best_k, "K Means (k={})".format(kmeans_best_k))
#
# base_experiment.describe_what_you_see(clfr.predict(x_train), y_train, plot_name,
#                                       4, "K Means (k={})".format(4))
#
# clfr = mixture.GaussianMixture(n_components=em_best_k, covariance_type='full')
# clfr.fit(x_train)
# base_experiment.describe_what_you_see(clfr.predict(x_train), y_train, plot_name,
#                                       kmeans_best_k, "EM (k={})".format(em_best_k))
#
# base_experiment.describe_what_you_see(clfr.predict(x_train), y_train, plot_name,
#                                       4, "EM (k={})".format(4))
#
# base_experiment.run_pca_and_plot(x_train, plot_name)
#
# base_experiment.run_ica_and_plot(x_train, plot_name, len(features))
#
# for k in range(1, 10):
#     rp = GaussianRandomProjection(n_components=k)
#
# base_experiment.run_random_projection_and_plot(x_train, plot_name, 25)
#
# base_experiment.selecting_features_with_low_variance(plot_name, features_data, features, 0.8)

# plot_points("{}:KMeans".format(plot_name), data[features].values, top_2_features)
# print(np.shape(y_train))
# print(np.shape(clfr.predict(x_train)))
#
