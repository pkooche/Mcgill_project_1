import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

#######################################
# Reading the dataset produced before
#######################################
dataset = pd.read_csv('datasets/merged_dataset.csv')

# Find different regions
regions = np.unique(dataset['open_covid_region_code'])

###################################
# Finding the most common dates
###################################
all_dates = []
for region in regions:
    dates = dataset[dataset['open_covid_region_code'] == region]['date'].values
    all_dates.append(set(dates))

final_dates = all_dates[0]

for date in all_dates:
    final_dates = final_dates.intersection(date)

final_dates = sorted(list(final_dates))
###############################################################################################################
# Generating an aggregated dataset in which the effect of each symptom is aggregated across different regions
###############################################################################################################

symptoms = [feature for feature in list(dataset.columns) if 'symptom' in feature]
columns = ['date']
columns.extend(symptoms)
aggregated_dataset = pd.DataFrame(columns=columns)

for date in final_dates:
    temp_df = pd.DataFrame(columns=symptoms)
    for region in regions:
        row = dataset[(dataset['open_covid_region_code'] == region) & (dataset['date'] == date)][symptoms]
        temp_df = temp_df.append(row, ignore_index=True)
    aggregated_values = list(temp_df.sum(axis=1).values)
    aggregated_values.insert(0, date)
    new_row = dict((k, v) for (k, v) in zip(columns, aggregated_values))
    aggregated_dataset = aggregated_dataset.append(new_row, ignore_index=True)

######################################
# Plotting the effect of each symptom
######################################

x_axis = aggregated_dataset['date'].values

for symptom in list(aggregated_dataset.columns):
    if symptom != 'date':
        y_axis = aggregated_dataset[symptom].values

        plt.plot(x_axis, y_axis, 'bo')
        plt.xlabel('Date')
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel(symptom)
        plt.title(f'Effect of "{symptom}" Aggregated Across All Regions')
        plt.show()
