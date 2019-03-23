import random
import sys

import numpy as np
import pandas as pd
from sklearn import mixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

random.seed(100)
np.random.seed(100)

best_k_for_kmeans = 4

best_k_for_em = 3


def plot_elbow_method_graph_kmeans(K, X, name):
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    import numpy as np
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt

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

train, test = train_test_split(data, test_size=0.2)

x_train = train[features]
y_train = train[target]

x_test = test[features]
y_test = test[target]
plot_name = "Phishing Detection"

# plot_elbow_method_graph_kmeans(range(2, 10), x_train)

plot_silhoutte_score_em(range(2, 10), x_train)

# kmeanModel = KMeans(n_clusters=4).fit(x_train)
clfr = mixture.GaussianMixture(n_components=4, covariance_type='full')
clfr.fit(x_train)
# kmeanModel.fit(x_train)

print(clfr.predict(x_train))
# print(y_train)
# print(kmeans.labels_ - y_train)

# ann_analysis(plot_name, data, features, target, x_train, y_train, x_test, y_test)
#
# plt.show()
