import random

import pandas as pd
import numpy as np


# Importing the dataset
dataset = pd.read_csv('datasets/merged_dataset.csv')

##############################################
# Train-Validation split based on timestamp
##############################################
regions = np.unique(dataset['open_covid_region_code'])

# Extracting the maximal interval of dates
dates = []

for region in regions:
    date = dataset[dataset['open_covid_region_code'] == region]['date'].values
    dates.append(set(date))

final_dates = dates[0]
for i in range(len(dates)):
    final_dates = final_dates.union(dates[i])

# final_dates contains the maximal range of dates available in the data
final_dates = sorted(list(final_dates))

# We set the last five dates as a threshold for splitting the data
threshold_date = final_dates[-5]

# Now we split the data into two sets; one set having dates past the threshold_date and another set having dates ahead
# the threshold date
training_set = dataset[dataset['date'] < threshold_date]
test_set = dataset[dataset['date'] >= threshold_date]

print('Dataset Shape', dataset.shape)
print('Training Set Shape :', training_set.shape)
print('Test Set Shape :', test_set.shape)

##############################################
# Train_Validation split based on regions
##############################################

# First, we choose five random regions to use their data as test set
test_regions = random.sample(list(regions), 5)

region_training_set = dataset[~dataset['open_covid_region_code'].isin(test_regions)]
region_test_set = dataset[dataset['open_covid_region_code'].isin(test_regions)]

