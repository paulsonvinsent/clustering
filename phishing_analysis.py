import sys

import pandas as pd
from sklearn import preprocessing

import base_experiment


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
features = data[features]

# top_2_features = base_experiment.identify_top_2_features(features)

x_train = preprocessing.scale(features)
y_train = data[target]

plot_name = "Phishing Detection"

base_experiment.plot_score_em(range(2, 20), x_train, plot_name)
base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), x_train, plot_name)
base_experiment.plot_silhoutte_score_kmeans(range(2, 20), x_train, plot_name)
base_experiment.plot_elbow_method_graph_kmeans(range(2, 20), x_train, plot_name)

# run_pca_and_plot(x_train, plot_name)
# print(np.shape(np.var(data[features], axis=0)))
# print(np.var(data[features], axis=0))
# print((np.var(data[features], axis=0)).argsort()[-2:])

# run_random_projection_and_plot(x_train,plot_name,len(features))

# plot_silhoutte_score_kmeans(range(2, 10), x_train, plot_name)
# clfr = KMeans(n_clusters=best_k_for_kmeans)
# clfr.fit(x_train)

# plot_points("{}:KMeans".format(plot_name), data[features].values, top_2_features, clfr.predict(x_train),
#             clfr.cluster_centers_,
#             best_k_for_kmeans)

# selecting_features_with_low_variance(plot_name, data[features], features)
#


# plot_points("{}:KMeans".format(plot_name), data[features].values, top_2_features)
# print(np.shape(y_train))
# print(np.shape(clfr.predict(x_train)))
#
# describe_what_you_see(clfr.predict(x_train), y_train, '{} - KMeans algorithm'.format(plot_name), x_train, features,
#                       best_k_for_kmeans)
