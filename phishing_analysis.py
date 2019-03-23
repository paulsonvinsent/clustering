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
    plt.title('{} : The Elbow Method showing the optimal k'.format(name))
    plt.show()


def plot_silhoutte_score_em(K, X, name):
    plt.plot()
    # k means determine k
    silhouette_scores = []
    for k in K:
        clfr = mixture.GaussianMixture(n_components=k, covariance_type='full')
        clfr.fit(X)
        silhouette_scores.append(silhouette_score(X, clfr.predict(X)))

    # Plot the elbow
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Scores')
    plt.title('{} : Silhouette Method showing the optimal k'.format(name))
    plt.show()


def describe_what_you_see(cluster_labels,
                          classification_labels, name,
                          X, features):
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
        y_labels = ['0', '1']
        plt.bar(y_pos, counts, align='center', alpha=0.5)
        plt.xticks(y_pos, y_labels)
        print(y_labels)
        plt.ylabel('Counts')
        plt.title('{} : Simple clustering results ,Cluster {} '.format(name, cluster))
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

x_train = data[features]
y_train = data[target]

plot_name = "Phishing Detection"

# plot_elbow_method_graph_kmeans(range(2, 10), x_train)

plot_silhoutte_score_em(range(2, 10), x_train,plot_name)

# kmeanModel = KMeans(n_clusters=4).fit(x_train)
clfr = mixture.GaussianMixture(n_components=best_k_for_em, covariance_type='full')
clfr.fit(x_train)
# kmeanModel.fit(x_train)

print(np.shape(y_train))
print(np.shape(clfr.predict(x_train)))


describe_what_you_see(clfr.predict(x_train), y_train, 'EM algorithm', x_train, features)

# print(y_train)
# print(kmeans.labels_ - y_train)

# ann_analysis(plot_name, data, features, target, x_train, y_train, x_test, y_test)
#
# plt.show()
