import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

##########################################
# Visualizing the Search Trends Dataset
##########################################

# Importing the search trends dataset
us_weekly = pd.read_csv('datasets/2020_US_weekly_symptoms_dataset.csv')

# Removing uninformative features
us_nan_values = us_weekly.isna().sum() / us_weekly.shape[0]
us_nan_values = us_nan_values[us_nan_values < 0.4]
us_proper_columns = list(us_nan_values.axes[0])
us_weekly = us_weekly[us_proper_columns]

# In order to visualize the data, first we need to aggregate the data of all regions on same dates
regions = np.unique(us_weekly['open_covid_region_code'])

# Finding the common dates
all_dates = []
for region in regions:
    dates = us_weekly[us_weekly['open_covid_region_code'] == region]['date'].values
    all_dates.append(set(dates))

final_dates = all_dates[0]

for date in all_dates:
    final_dates = final_dates.intersection(date)

final_dates = sorted(list(final_dates))
################################
# Aggregating the symptoms
symptoms = [feature for feature in list(us_weekly.columns) if 'symptom' in feature]
columns = ['date']
columns.extend(symptoms)
aggregated_dataset = pd.DataFrame(columns=columns)

for date in final_dates:
    temp_df = pd.DataFrame(columns=symptoms)
    for region in regions:
        row = us_weekly[(us_weekly['open_covid_region_code'] == region) & (us_weekly['date'] == date)][symptoms]
        temp_df = temp_df.append(row, ignore_index=True)
    aggregated_values = list(temp_df.sum(axis=1).values)
    aggregated_values.insert(0, date)
    new_row = dict((k, v) for (k, v) in zip(columns, aggregated_values))
    aggregated_dataset = aggregated_dataset.append(new_row, ignore_index=True)

original_data = aggregated_dataset[symptoms].values
##################################
# Dimensionality reduction using PCA
# number of component can be 2 or 3 for better visualization

# 3D visualization
number_of_components = 3
pca = PCA(n_components=number_of_components)
transformed_data_3d = pca.fit_transform(original_data)
print(original_data.shape, transformed_data_3d.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(transformed_data_3d[:, 0], transformed_data_3d[:, 1], transformed_data_3d[:, 2])
plt.show()

# 2D visualization
number_of_components = 2
pca = PCA(n_components=number_of_components)
transformed_data_2d = pca.fit_transform(original_data)
print(original_data.shape, transformed_data_2d.shape)

plt.scatter(transformed_data_2d[:, 0], transformed_data_2d[:, 1])
plt.show()

###################################
# K-Means clustering on both original data and transformed data
# The results show that cluster labels for data points varies for original data and dimension-reduced data
# Reducing the number of clusters leads to fewer changes in cluster labels

number_of_clusters = 5

clusterer = KMeans(n_clusters=number_of_clusters)

clusterer.fit(original_data)
print('Cluster Labels For Original Data: ', clusterer.labels_)

clusterer.fit(transformed_data_2d)
print('Cluster Labels For 2D Data: ', clusterer.labels_)

clusterer.fit(transformed_data_3d)
print('Cluster Labels For 3D Data: ', clusterer.labels_)

