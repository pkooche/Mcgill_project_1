import pandas as pd
import numpy as np


################################################
# Reading the datasets into pandas Data Frame
################################################

us_weekly = pd.read_csv('datasets/2020_US_weekly_symptoms_dataset.csv')
aggregated_cc = pd.read_csv('datasets/aggregated_cc_by.csv', low_memory=False)

# Getting the size of datasets
print('2020 US Weekly Symptoms Dataset Size :', us_weekly.shape)
print('Aggregated CC By Dataset Size :', aggregated_cc.shape)

##########################################
# Filtering aggregated_cc for US regions
##########################################

condition = aggregated_cc['open_covid_region_code'].str.contains('US-')
aggregated_cc = aggregated_cc[condition]

print('Filtered Aggregated CC By Dataset Size :', aggregated_cc.shape)

#####################
# Cleaning the data
#####################

# AGGREGATED_CC dataset
################################

# Percentage of NaN values in each column
aggregated_nan_values = aggregated_cc.isna().sum() / aggregated_cc.shape[0]

# Show the columns whose number of NaN values are less than 60 percent
# Only 'open_covid_region_code', 'region_name', 'date', 'hospitalized_new', and 'hospitalized_cumulative' columns
# have sufficient non-Nan values
aggregated_nan_values = aggregated_nan_values[aggregated_nan_values < 0.6]
aggregated_proper_columns = list(aggregated_nan_values.axes[0])
# Getting the features that are not appropriate
aggregated_non_proper_features = [feature for feature in aggregated_cc.columns if feature not in
                                  aggregated_proper_columns]
print('Non-proper features : ', aggregated_non_proper_features)

# Keeping the appropriate columns
aggregated_cc = aggregated_cc[aggregated_proper_columns]
print('Filtered Aggregated CC By Dataset Size :', aggregated_cc.shape)

# US_WEEKLY dataset
################################

us_nan_values = us_weekly.isna().sum() / us_weekly.shape[0]

# Show the columns whose number of NaN values are less than 40 percent
us_nan_values = us_nan_values[us_nan_values < 0.4]
us_proper_columns = list(us_nan_values.axes[0])
# Getting the features that are not appropriate
us_non_proper_features = [feature for feature in us_weekly.columns if feature not in us_proper_columns]
print('Non-proper features : ', us_non_proper_features)

# Keeping the appropriate columns
us_weekly = us_weekly[us_proper_columns]
print('Filtered US Weekly Dataset Size :', us_weekly.shape)

############################
# Merging the two datasets
############################

new_aggregated_cc = pd.DataFrame(columns=aggregated_cc.columns)
regions = np.unique(aggregated_cc['open_covid_region_code'])

for region in regions:
    data = aggregated_cc[aggregated_cc['open_covid_region_code'] == region]

    # Fining the first common date in both datasets for a given region. It is necessary for making the aggregated_cc
    # dataset into a weekly format
    agg_dates = set(data['date'])
    us_dates = set(us_weekly[us_weekly['open_covid_region_code'] == region]['date'])

    # We only find the first common date if the region appears in both datasets. We omit the regions that are present
    # in only one dataset
    if us_dates == set():
        continue
    else:
        first_common_date = sorted(list(agg_dates.intersection(us_dates)))[0]
        first_common_date_index = np.where(data['date'] == first_common_date)[0][0]

        if first_common_date_index < 6:
            first_common_date = sorted(list(agg_dates.intersection(us_dates)))[1]
            first_common_date_index = np.where(data['date'] == first_common_date)[0][0]

    for i in range(first_common_date_index - 6, len(data) - 6, 7):
        date = data.iloc[i + 6]['date']
        hospitalized_new = 0

        for j in range(7):
            hospitalized_new += data.iloc[i + j]['hospitalized_new']

        hospitalized_cumulative = hospitalized_new - data.iloc[i]['hospitalized_new'] + data.iloc[i]['hospitalized_cumulative']
        new_data = {'open_covid_region_code': region,
                    'region_name': data.iloc[i]['region_name'],
                    'date': date,
                    'hospitalized_new': hospitalized_new,
                    'hospitalized_cumulative': hospitalized_cumulative}

        new_aggregated_cc = new_aggregated_cc.append(new_data, ignore_index=True)

common_regions = np.unique(new_aggregated_cc['open_covid_region_code'])
new_columns = list(new_aggregated_cc.columns)
agg_query_columns = list(new_aggregated_cc.columns).copy()
agg_query_columns.remove('open_covid_region_code')
us_query_columns = list(us_weekly.columns).copy()
us_query_columns.remove('open_covid_region_code')

for column in us_weekly.columns:
    if column not in new_columns:
        new_columns.append(column)

merged_dataset = pd.DataFrame(columns=new_columns)

for region in common_regions:
    agg_data = new_aggregated_cc[new_aggregated_cc['open_covid_region_code'] == region][agg_query_columns]
    us_data = us_weekly[us_weekly['open_covid_region_code'] == region][us_query_columns]

    merged = pd.merge(agg_data, us_data, how='inner', on='date')
    merged.insert(0, 'open_covid_region_code', region)
    merged_dataset = merged_dataset.append(merged)

merged_dataset = merged_dataset.reset_index()
merged_dataset = merged_dataset.drop(columns=['index'])
print(merged_dataset)

merged_dataset.to_csv('datasets/merged_dataset.csv', index=False)
