import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.svm import SVR

pd.options.mode.chained_assignment = None

# In order to perform a 5-fold cross-validation with the two strategies introduced before, first we need to implement
# a custom 5-fold cross-validation. We also write a custom function, 'impute' to fill the NaN values. The strategy for
# filling the NaN values is to fill each NaN value with the mean of last three days


def cross_validation(ds, n_folds=5, mode='region'):
    """
    :param ds: the covid dataset
    :param n_folds: number of folds for cross validation
    :param mode: 'region' or 'date'. splits the data based on region or date
    :return: train and test sets. Each set contains a list with n_fold items. Each item is the splitted dataset
    """
    # Finding the unique regions in the dataset
    regions = np.unique(ds['open_covid_region_code'])

    # Finding the date span across the dataset
    dates = []

    for region in regions:
        date = ds[ds['open_covid_region_code'] == region]['date'].values
        dates.append(set(date))

    final_dates = dates[0]
    for i in range(len(dates)):
        final_dates = final_dates.union(dates[i])

    dates = sorted(list(final_dates))
    #########################################
    if mode == 'region':
        number_of_regions = len(regions)

        folds_size = int(number_of_regions / n_folds)

        train_sets = []
        validation_sets = []

        for i in range(n_folds):
            regions_copy = list(regions).copy()

            if i == n_folds - 1:
                validation_regions = regions_copy[-folds_size:]
                del regions_copy[-folds_size:]
            else:
                validation_regions = regions_copy[i * folds_size: (i + 1) * folds_size]
                del regions_copy[i * folds_size: (i + 1) * folds_size]

            train_regions = regions_copy

            validation_set = ds[ds['open_covid_region_code'].isin(validation_regions)]
            train_set = ds[ds['open_covid_region_code'].isin(train_regions)]

            validation_sets.append(validation_set)
            train_sets.append(train_set)

        return train_sets, validation_sets
    elif mode == 'date':
        number_of_dates = len(dates)

        folds_size = int(number_of_dates / n_folds)

        train_sets = []
        validations_sets = []

        for i in range(n_folds):
            dates_copy = list(dates).copy()

            if i == n_folds - 1:
                validation_dates = dates_copy[-folds_size:]
                del dates_copy[-folds_size:]
            else:
                validation_dates = dates_copy[i * folds_size: (i + 1) * folds_size]
                del dates_copy[i * folds_size: (i + 1) * folds_size]

            train_dates = dates_copy

            validations_set = ds[ds['date'].isin(validation_dates)]
            train_set = ds[ds['date'].isin(train_dates)]

            validations_sets.append(validations_set)
            train_sets.append(train_set)

        return train_sets, validations_sets


def impute(ds, mode='mean'):
    """
    This function fills the nan values of the dataset. This is necessary for regression task
    :param ds: The covid dataset
    :param mode: 'mean', 'zero', 'g_mean', and 'knn'. 'mean' mode fills the nan values based on weighted average of the
    past three days. 'zero' mode fills the nan values with 0. 'g_mean' fills the nan values with the total mean of the
    column. 'knn' mode fills the nan values based on the average of the k nearest neighbors.
    :return: None. Updates the dataset to a dataset with no nan values.
    """
    # Find which columns has Nan values
    nan_columns = ds.isna().any()

    features = list(nan_columns.axes[0])

    if mode == 'mean':
        for feature in features:
            if nan_columns[feature]:
                for i in range(len(ds[feature])):
                    if pd.isnull(ds[feature].iloc[i]):
                        if i == 0:
                            j = i
                            while pd.isnull(ds[feature].iloc[j]):
                                j += 1
                            for k in range(j):
                                ds[feature].iloc[k] = ds[feature].iloc[j]
                        elif i == 1:
                            ds[feature].iloc[i] = ds[feature].iloc[i - 1]
                        elif i == 2:
                            ds[feature].iloc[i] = 0.5 * (2 * ds[feature].iloc[i - 1] + ds[feature].iloc[i - 2])
                        else:
                            ds[feature].iloc[i] = (ds[feature].iloc[i - 1] + 0.7 * ds[feature].iloc[i - 2] +
                                                   0.5 * ds[feature].iloc[i - 3]) / 3
                    else:
                        continue
    elif mode == 'zero':
        for feature in features:
            if nan_columns[feature]:
                for i in range(len(ds[feature])):
                    if pd.isnull(ds[feature].iloc[i]):
                        ds[feature].iloc[i] = 0.
    elif mode == 'g_mean':
        imputer = SimpleImputer(strategy='mean')

        for feature in features:
            if nan_columns[feature]:
                ds[feature] = imputer.fit_transform(ds[feature].values.reshape(-1, 1))
    elif mode == 'knn':
        imputer = KNNImputer(n_neighbors=2)

        for feature in features:
            if nan_columns[feature]:
                ds[feature] = imputer.fit_transform(ds[feature].values.reshape(-1, 1))


# Reading the dataset
dataset = pd.read_csv('datasets/merged_dataset.csv')

# Change open_covid_region_code and date into a Categorical column
dataset['open_covid_region_code'] = dataset['open_covid_region_code'].astype('category')
dataset['date'] = dataset['date'].astype('category')

cat_columns = dataset.select_dtypes(['category']).columns

dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)

# Imputing the NaN values
impute(dataset, mode='mean')

# Generating train sets and validations sets based on 5-fold cross validation
train, validation = cross_validation(dataset, mode='date')

# It is better to drop columns {region_name, country_region_code, country_region, sub_region_1, sub_region_1_code}
# since they do not provide any further information. All of their information is encapsulated in open_covid_region_code
# feature
print('KNN for data folded by date')
results = []
for i_train_set, i_validation_set in zip(train, validation):
    new_train_set = i_train_set.drop(columns=['region_name', 'country_region_code', 'country_region',
                                              'sub_region_1', 'sub_region_1_code'])
    new_validation_set = i_validation_set.drop(columns=['region_name', 'country_region_code', 'country_region',
                                                        'sub_region_1', 'sub_region_1_code'])

    y_column = 'hospitalized_new'
    x_columns = list(new_train_set.columns)
    x_columns.remove(y_column)
    # We remove the feature below because it has full correlation with hospitalized new, which we need to predict
    x_columns.remove('hospitalized_cumulative')

    x_values = new_train_set[x_columns]
    y_values = new_train_set[y_column]

    x_validation = new_validation_set[x_columns]
    y_validation = new_validation_set[y_column]

    knn_regressor = KNeighborsRegressor(n_neighbors=3)
    knn_regressor.fit(x_values, y_values)
    train_predictions = knn_regressor.predict(x_values)
    validation_predictions = knn_regressor.predict(x_validation)
    train_error = mean_squared_error(y_values, train_predictions)
    test_error = mean_squared_error(y_validation, validation_predictions)
    train_score = knn_regressor.score(x_values, y_values)
    test_score = knn_regressor.score(x_validation, y_validation)

    results.append([train_error, test_error, train_score, test_score])

print(max(results, key=lambda x: x[-1]))

#################################################################################
# Generating train sets and validations sets based on 5-fold cross validation
train, validation = cross_validation(dataset, mode='region')

# It is better to drop columns {region_name, country_region_code, country_region, sub_region_1, sub_region_1_code}
# since they do not provide any further information. All of their information is encapsulated in open_covid_region_code
# feature
print('KNN for data folded by region')
results = []
for i_train_set, i_validation_set in zip(train, validation):
    new_train_set = i_train_set.drop(columns=['region_name', 'country_region_code', 'country_region',
                                              'sub_region_1', 'sub_region_1_code'])
    new_validation_set = i_validation_set.drop(columns=['region_name', 'country_region_code', 'country_region',
                                                        'sub_region_1', 'sub_region_1_code'])

    y_column = 'hospitalized_new'
    x_columns = list(new_train_set.columns)
    x_columns.remove(y_column)
    # We remove the feature below because it has full correlation with hospitalized new, which we need to predict
    x_columns.remove('hospitalized_cumulative')

    x_values = new_train_set[x_columns]
    y_values = new_train_set[y_column]

    x_validation = new_validation_set[x_columns]
    y_validation = new_validation_set[y_column]

    knn_regressor = KNeighborsRegressor(n_neighbors=3)
    knn_regressor.fit(x_values, y_values)
    train_predictions = knn_regressor.predict(x_values)
    validation_predictions = knn_regressor.predict(x_validation)
    train_error = mean_squared_error(y_values, train_predictions)
    test_error = mean_squared_error(y_validation, validation_predictions)
    train_score = knn_regressor.score(x_values, y_values)
    test_score = knn_regressor.score(x_validation, y_validation)

    results.append([train_error, test_error, train_score, test_score])

print(max(results, key=lambda x: x[-1]))

##################################################################################
# Generating train sets and validations sets based on 5-fold cross validation
train, validation = cross_validation(dataset, mode='date')

# It is better to drop columns {region_name, country_region_code, country_region, sub_region_1, sub_region_1_code}
# since they do not provide any further information. All of their information is encapsulated in open_covid_region_code
# feature
print('Decision Tree for data folded by date')
results = []
for i_train_set, i_validation_set in zip(train, validation):
    new_train_set = i_train_set.drop(columns=['region_name', 'country_region_code', 'country_region',
                                              'sub_region_1', 'sub_region_1_code'])
    new_validation_set = i_validation_set.drop(columns=['region_name', 'country_region_code', 'country_region',
                                                        'sub_region_1', 'sub_region_1_code'])

    y_column = 'hospitalized_new'
    x_columns = list(new_train_set.columns)
    x_columns.remove(y_column)
    # We remove the feature below because it has full correlation with hospitalized new, which we need to predict
    x_columns.remove('hospitalized_cumulative')

    x_values = new_train_set[x_columns]
    y_values = new_train_set[y_column]

    x_validation = new_validation_set[x_columns]
    y_validation = new_validation_set[y_column]

    dt_regressor = DecisionTreeRegressor(random_state=0)
    dt_regressor.fit(x_values, y_values)
    train_predictions = dt_regressor.predict(x_values)
    validation_predictions = dt_regressor.predict(x_validation)
    train_error = mean_squared_error(y_values, train_predictions)
    test_error = mean_squared_error(y_validation, validation_predictions)
    train_score = dt_regressor.score(x_values, y_values)
    test_score = dt_regressor.score(x_validation, y_validation)

    results.append([train_error, test_error, train_score, test_score])

print(max(results, key=lambda x: x[-1]))

####################################################################################
# Generating train sets and validations sets based on 5-fold cross validation
train, validation = cross_validation(dataset, mode='region')

# It is better to drop columns {region_name, country_region_code, country_region, sub_region_1, sub_region_1_code}
# since they do not provide any further information. All of their information is encapsulated in open_covid_region_code
# feature
print('Decision Tree for data folded by region')
results = []
for i_train_set, i_validation_set in zip(train, validation):
    new_train_set = i_train_set.drop(columns=['region_name', 'country_region_code', 'country_region',
                                              'sub_region_1', 'sub_region_1_code'])
    new_validation_set = i_validation_set.drop(columns=['region_name', 'country_region_code', 'country_region',
                                                        'sub_region_1', 'sub_region_1_code'])

    y_column = 'hospitalized_new'
    x_columns = list(new_train_set.columns)
    x_columns.remove(y_column)
    # We remove the feature below because it has full correlation with hospitalized new, which we need to predict
    x_columns.remove('hospitalized_cumulative')

    x_values = new_train_set[x_columns]
    y_values = new_train_set[y_column]

    x_validation = new_validation_set[x_columns]
    y_validation = new_validation_set[y_column]

    dt_regressor = DecisionTreeRegressor(random_state=0)
    dt_regressor.fit(x_values, y_values)
    train_predictions = dt_regressor.predict(x_values)
    validation_predictions = dt_regressor.predict(x_validation)
    train_error = mean_squared_error(y_values, train_predictions)
    test_error = mean_squared_error(y_validation, validation_predictions)
    train_score = dt_regressor.score(x_values, y_values)
    test_score = dt_regressor.score(x_validation, y_validation)

    results.append([train_error, test_error, train_score, test_score])

print(max(results, key=lambda x: x[-1]))

################################################################################################################

############################################################
# Exploring the effect of Support Vector Machine Regression
############################################################

# Generating train sets and validations sets based on 5-fold cross validation
train, validation = cross_validation(dataset, mode='date')

# It is better to drop columns {region_name, country_region_code, country_region, sub_region_1, sub_region_1_code}
# since they do not provide any further information. All of their information is encapsulated in open_covid_region_code
# feature
print('SVR for data folded by date')
results = []
for i_train_set, i_validation_set in zip(train, validation):
    new_train_set = i_train_set.drop(columns=['region_name', 'country_region_code', 'country_region',
                                              'sub_region_1', 'sub_region_1_code'])
    new_validation_set = i_validation_set.drop(columns=['region_name', 'country_region_code', 'country_region',
                                                        'sub_region_1', 'sub_region_1_code'])

    y_column = 'hospitalized_new'
    x_columns = list(new_train_set.columns)
    x_columns.remove(y_column)
    # We remove the feature below because it has full correlation with hospitalized new, which we need to predict
    x_columns.remove('hospitalized_cumulative')

    x_values = new_train_set[x_columns]
    y_values = new_train_set[y_column]

    x_validation = new_validation_set[x_columns]
    y_validation = new_validation_set[y_column]

    svr = SVR(C=300)
    svr.fit(x_values, y_values)
    train_predictions = svr.predict(x_values)
    validation_predictions = svr.predict(x_validation)
    train_error = mean_squared_error(y_values, train_predictions)
    test_error = mean_squared_error(y_validation, validation_predictions)
    train_score = svr.score(x_values, y_values)
    test_score = svr.score(x_validation, y_validation)

    results.append([train_error, test_error, train_score, test_score])

print(max(results, key=lambda x: x[-1]))

###################################################################################

# Generating train sets and validations sets based on 5-fold cross validation
train, validation = cross_validation(dataset, mode='region')

# It is better to drop columns {region_name, country_region_code, country_region, sub_region_1, sub_region_1_code}
# since they do not provide any further information. All of their information is encapsulated in open_covid_region_code
# feature
print('SVR for data folded by region')
results = []
for i_train_set, i_validation_set in zip(train, validation):
    new_train_set = i_train_set.drop(columns=['region_name', 'country_region_code', 'country_region',
                                              'sub_region_1', 'sub_region_1_code'])
    new_validation_set = i_validation_set.drop(columns=['region_name', 'country_region_code', 'country_region',
                                                        'sub_region_1', 'sub_region_1_code'])

    y_column = 'hospitalized_new'
    x_columns = list(new_train_set.columns)
    x_columns.remove(y_column)
    # We remove the feature below because it has full correlation with hospitalized new, which we need to predict
    x_columns.remove('hospitalized_cumulative')

    x_values = new_train_set[x_columns]
    y_values = new_train_set[y_column]

    x_validation = new_validation_set[x_columns]
    y_validation = new_validation_set[y_column]

    svr = SVR(C=100)
    svr.fit(x_values, y_values)
    train_predictions = svr.predict(x_values)
    validation_predictions = svr.predict(x_validation)
    train_error = mean_squared_error(y_values, train_predictions)
    test_error = mean_squared_error(y_validation, validation_predictions)
    train_score = svr.score(x_values, y_values)
    test_score = svr.score(x_validation, y_validation)

    results.append([train_error, test_error, train_score, test_score])

print(max(results, key=lambda x: x[-1]))

##################################################################################################################
# From the results obtained above we can conclude that the best results can be achieved by Decision Tree and Support
# Vector Machines and when split the data using the date information
##################################################################################################################


