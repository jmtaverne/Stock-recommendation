from os import listdir
from os.path import isfile, join, normpath
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
from sklearn import preprocessing
from pandas import DataFrame
from pandas import concat
import datetime
from datetime import timedelta, date
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV,  cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import csv
import time
import multiprocessing as mp

# GLOBALS

boersen_days_2018 = [
    "2018-01-02",
    "2018-01-03",
    "2018-01-04",
    "2018-01-05",
    "2018-01-08",
    "2018-01-09",
    "2018-01-10",
    "2018-01-11",
    "2018-01-12",
    "2018-01-16",
    "2018-01-17",
    "2018-01-18",
    "2018-01-19",
    "2018-01-22",
    "2018-01-23",
    "2018-01-24",
    "2018-01-25",
    "2018-01-26",
    "2018-01-29",
    "2018-01-30",
    "2018-01-31",
    "2018-02-01",
    "2018-02-02",
    "2018-02-05",
    "2018-02-06",
    "2018-02-07",
    "2018-02-08",
    "2018-02-09",
    "2018-02-12",
    "2018-02-13",
    "2018-02-14",
    "2018-02-15",
    "2018-02-16",
    "2018-02-20",
    "2018-02-21",
    "2018-02-22",
    "2018-02-23",
    "2018-02-26",
    "2018-02-27",
    "2018-02-28",
    "2018-03-01",
    "2018-03-02",
    "2018-03-05",
    "2018-03-06",
    "2018-03-07",
    "2018-03-08",
    "2018-03-09",
    "2018-03-12",
    "2018-03-13",
    "2018-03-14",
    "2018-03-15",
    "2018-03-16",
    "2018-03-19",
    "2018-03-20",
    "2018-03-21",
    "2018-03-22",
    "2018-03-23",
    "2018-03-26",
    "2018-03-27",
    "2018-03-28",
    "2018-03-29",
    "2018-04-02",
    "2018-04-03",
    "2018-04-04",
    "2018-04-05",
    "2018-04-06",
    "2018-04-09",
    "2018-04-10",
    "2018-04-11",
    "2018-04-12",
    "2018-04-13",
    "2018-04-16",
    "2018-04-17",
    "2018-04-18",
    "2018-04-19",
    "2018-04-20",
    "2018-04-23",
    "2018-04-24",
    "2018-04-25",
    "2018-04-26",
    "2018-04-27",
    "2018-04-30",
    "2018-05-01",
    "2018-05-02",
    "2018-05-03",
    "2018-05-04",
    "2018-05-07",
    "2018-05-08",
    "2018-05-09",
    "2018-05-10",
    "2018-05-11",
    "2018-05-14",
    "2018-05-15",
    "2018-05-16",
    "2018-05-17",
    "2018-05-18",
    "2018-05-21",
    "2018-05-22",
    "2018-05-23",
    "2018-05-24",
    "2018-05-25",
    "2018-05-29",
    "2018-05-30",
    "2018-05-31",
    "2018-06-01",
    "2018-06-04",
    "2018-06-05",
    "2018-06-06",
    "2018-06-07",
    "2018-06-08",
    "2018-06-11",
    "2018-06-12",
    "2018-06-13",
    "2018-06-14",
    "2018-06-15",
    "2018-06-18",
    "2018-06-19",
    "2018-06-20",
    "2018-06-21",
    "2018-06-22",
    "2018-06-25",
    "2018-06-26",
    "2018-06-27",
    "2018-06-28",
    "2018-06-29"
]

boersen_days_2017 = [
    "2017-08-23",
    "2017-08-24",
    "2017-08-25",
    "2017-08-28",
    "2017-08-29",
    "2017-08-30",
    "2017-08-31",
    "2017-09-01",
    "2017-09-05",
    "2017-09-06",
    "2017-09-07",
    "2017-09-08",
    "2017-09-11",
    "2017-09-12",
    "2017-09-13",
    "2017-09-14",
    "2017-09-15",
    "2017-09-18",
    "2017-09-19",
    "2017-09-20",
    "2017-09-21",
    "2017-09-22",
    "2017-09-25",
    "2017-09-26",
    "2017-09-27",
    "2017-09-28",
    "2017-09-29",
    "2017-10-02",
    "2017-10-03",
    "2017-10-04",
    "2017-10-05",
    "2017-10-06",
    "2017-10-09",
    "2017-10-10",
    "2017-10-11",
    "2017-10-12",
    "2017-10-13",
    "2017-10-16",
    "2017-10-17",
    "2017-10-18",
    "2017-10-19",
    "2017-10-20",
    "2017-10-23",
    "2017-10-24",
    "2017-10-25",
    "2017-10-26",
    "2017-10-27",
    "2017-10-30",
    "2017-10-31",
    "2017-11-01",
    "2017-11-02",
    "2017-11-03",
    "2017-11-06",
    "2017-11-07",
    "2017-11-08",
    "2017-11-09",
    "2017-11-10",
    "2017-11-13",
    "2017-11-14",
    "2017-11-15",
    "2017-11-16",
    "2017-11-17",
    "2017-11-20",
    "2017-11-21",
    "2017-11-22",
    "2017-11-24",
    "2017-11-27",
    "2017-11-28",
    "2017-11-29",
    "2017-11-30",
    "2017-12-01",
    "2017-12-04",
    "2017-12-05",
    "2017-12-06",
    "2017-12-07",
    "2017-12-08",
    "2017-12-11",
    "2017-12-12",
    "2017-12-13",
    "2017-12-14",
    "2017-12-15",
    "2017-12-18",
    "2017-12-19",
    "2017-12-20",
    "2017-12-21",
    "2017-12-22",
    "2017-12-26",
    "2017-12-27",
    "2017-12-28",
    "2017-12-29"
]

all_boersen_days = boersen_days_2017
all_boersen_days.extend(boersen_days_2018)

features_time_range = [i*10 for i in range(10)]
features_used = ['Open', 'Close', 'Low', 'High', 'Volume']

dir_path = "C:/Users/Katja/Desktop/Big Data/Projekt/data/"
dir_path = normpath(dir_path)

# ticker file names
ticker_path = join(dir_path, "stocks")
ticker_file_names = [f for f in listdir(ticker_path) if (isfile(join(ticker_path, f)) and ".csv" in f)]

# FUNCTIONS

def series_to_supervised(feature_name, data, time_range=[0], dropnan=True):
    df = DataFrame(data)
    cols, names = list(), list()
    # forecast sequence (t, t+1, ... t+n)
    for i in time_range:
        cols.append(df.shift(-i))
        names += [feature_name + "_" + str(i)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def hasTickerAllDates(ticker_df):
    res = True
    for date_str in all_boersen_days:
        if not date_str in ticker_df['Date'].values:
            res = False
            continue
    return res


def getFeatureTimeseriesDf(feature, feature_df):
    timeseries_feature_df = series_to_supervised(feature, feature_df[feature].values, features_time_range)

    feature_ = feature + "_"

    for t in features_time_range:
        if not t == features_time_range[0]:
            f_0 = timeseries_feature_df[feature_ + str(features_time_range[0])]
            f_t = timeseries_feature_df[feature_ + str(t)]
            f_t = f_t - f_0
            feature_df[feature_ + str(t)] = f_t
    if len(features_time_range) > 1:
        feature_df = feature_df.drop([feature], axis=1)
    return feature_df


def normalizeFeatureDf(feature_df):
    tmp_df = feature_df.copy(deep=True)
    tmp_df = tmp_df.drop(["Date"], axis=1)
    scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)
    scaler.fit_transform(tmp_df.values)
    tmp_df["Date"] = feature_df["Date"].values
    feature_df = tmp_df

    # move the "Date" column to the front of the dataframe
    cols = list(feature_df)
    cols.insert(0, cols.pop(cols.index('Date')))
    feature_df = feature_df.loc[:, cols]
    return feature_df


def getTickerTimeseriesDf(df_ticker):
    stock_merged_df = DataFrame()
    first_feature_b = True

    for feature in features_used:
        feature_df = df_ticker.copy(deep=True)
        # drop everything except of the current feature and the date
        feature_df = feature_df[["Date", feature]]

        feature_df = getFeatureTimeseriesDf(feature, feature_df)

        feature_df = normalizeFeatureDf(feature_df)

        feature_df = feature_df.dropna(axis = 0)
        feature_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, inplace=True)

        # merge current dataframe into overall dataframe
        if first_feature_b:
            stock_merged_df = feature_df
            first_feature_b = False
        else:
            stock_merged_df = pd.merge(stock_merged_df, feature_df, on='Date')
    return stock_merged_df


def getTickerTimeseries_GT_Df(file_name, df_train_label):
    ticker_name = file_name.replace(".csv", "")

    print("Loading: " + file_name)

    # Load data frame
    file_str = join(ticker_path, file_name)
    df_ticker = pd.read_csv(file_str)
    df_ticker.columns = ['Date', 'Open', 'Close', 'Low', 'High', 'Volume']

    # if ticker is not valid, then write all zeros in the output file
    ticker_data_valid = hasTickerAllDates(df_ticker)
    if not ticker_data_valid:
        return None

    stock_merged_df = getTickerTimeseriesDf(df_ticker)

    stock_merged_df = stock_merged_df.dropna(axis = 0)
    stock_merged_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, inplace=True)

    # Add dependend variable
    labels_df = df_train_label.loc[:, df_train_label.columns.intersection([ticker_name])]
    labels_df = labels_df.rename(index=str, columns={ticker_name: 'Y'})
    stock_complete_df = pd.merge(stock_merged_df, labels_df[['Y']], on='Date')
    stock_complete_df = stock_complete_df.sort_values('Date')

    return stock_complete_df


def getAllTickerTimeseries_GT_Df_serial(ticker_file_names, df_train_label):
    all_stocks_df = DataFrame()

    first_stock_b = True
    file_ctr = 0

    num_samples = 50
    ticker_file_names = [ticker_file_names[i] for i in np.random.randint(0, len(ticker_file_names), num_samples)]

    for file_name in ticker_file_names:

        ticker_name = file_name.replace(".csv", "")

        file_ctr += 1

        stock_complete_df = getTickerTimeseries_GT_Df(file_name, df_train_label)

        if stock_complete_df is not None:
            if first_stock_b:
                all_stocks_df = stock_complete_df
                first_stock_b = False
            else:
                all_stocks_df = all_stocks_df.append(stock_complete_df, ignore_index=True)
                all_stocks_df = all_stocks_df.sort_values('Date')
        else:
            print(ticker_name + " not complete")
    return all_stocks_df

def getAllTickerTimeseries_GT_Df_multiprocess(ticker_file_names, df_train_label):
    pool = mp.Pool(processes=mp.cpu_count())

    res = [pool.apply(getTickerTimeseries_GT_Df, args=(file_name, df_train_label)) for file_name in ticker_file_names]

    all_tickers_df = DataFrame()
    first_stock_b = True

    for ticker_df in res:
        if ticker_df is not None:
                if first_stock_b:
                    all_tickers_df = ticker_df
                    first_stock_b = False
                else:
                    all_tickers_df = all_tickers_df.append(ticker_df, ignore_index=True)
                    all_tickers_df = all_tickers_df.sort_values('Date')

    return all_tickers_df

def prepareTrainingData():
    # load training labels
    df_train_label = pd.read_csv(join(dir_path, 'labels_train.csv'), header=0, index_col=0)

    start_time = time.time()

    all_stocks_df = getAllTickerTimeseries_GT_Df_serial(ticker_file_names, df_train_label)
    # all_stocks_df = getAllTickerTimeseries_GT_Df_multiprocess(ticker_file_names, df_train_label)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Ladedauer: ' + "{0:.2f}".format(elapsed_time) + ' s', end='\n')
    print()

    all_stocks_df = all_stocks_df.dropna(axis = 0)
    all_stocks_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, inplace=True)

    cols_x = list(all_stocks_df)
    cols_x.remove("Date")
    cols_x.remove("Y")

    x = all_stocks_df[cols_x]
    y = all_stocks_df["Y"]

    return x, y

def gridSearch(x, y):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.5, test_size=0.5, random_state=42)

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, n_jobs=-1, scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

def train(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=42)

    # Train Classifier
    print("Starting Training of SVM")
    print()
    # classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    # classifier = KNeighborsClassifier(n_neighbors=5)
    classifier = svm.SVC(kernel= 'rbf', gamma=0.001, C=100.0, max_iter=-1)

    classifier.fit(x_train, y_train)

    rf_score = classifier.score(x_test, y_test)
    print("SVM Score: " + ": " + str(rf_score))

    return classifier

def classify_2018(classifier):
    result_str_lst = list()

    for file_name in ticker_file_names:

        ticker_name = file_name.replace(".csv", "")
        print("Predicting " + ticker_name)

        # 1 Load data frame
        file_str = join(ticker_path, file_name)
        df_ticker = pd.read_csv(file_str)
        df_ticker.columns = ['Date', 'Open', 'Close', 'Low', 'High', 'Volume']

        ticker_data_valid = hasTickerAllDates(df_ticker)

        # if ticker is not valid, then write all zeros in the output file
        if not ticker_data_valid:
            for date_str in boersen_days_2018:
                result_str_lst.append([date_str + ":" + ticker_name, 0])
            print(ticker_name + " defaults to 0")
            continue

        ticker_ts_df = getTickerTimeseriesDf(df_ticker)

        ticker_ts_df = ticker_ts_df.dropna(axis = 0)
        ticker_ts_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, inplace=True)


        selected_dates_df = DataFrame()
        selected_dates = list()
        default_dates = list()

        for date in boersen_days_2018:
            x = ticker_ts_df.loc[ticker_ts_df['Date'] == date]
            if not x.empty:
                selected_dates_df = selected_dates_df.append(x)
                selected_dates.append(date)
            else:
                default_dates.append(date)

        selected_dates_df = selected_dates_df.drop(['Date'], axis=1)

        prediction = classifier.predict(selected_dates_df)

        for _ in range(len(prediction) + len(default_dates)):
            if len(selected_dates) > 0:
                min_pred = min(selected_dates)
            else:
                min_pred = '9999-99-99'
            if len(default_dates) > 0:
                min_default = min(default_dates)
            else:
                min_default = '9999-99-99'

            if min_pred < min_default:
                min_pred_idx = selected_dates.index(min_pred)
                result_str_lst.append(
                    [min_pred + ":" + ticker_name, prediction[min_pred_idx]])
                del selected_dates[min_pred_idx]
                prediction = np.delete(prediction, min_pred_idx)
            else:
                min_default_idx = default_dates.index(min_default)
                result_str_lst.append(
                    [min_default + ":" + ticker_name, 0])
                del default_dates[min_default_idx]
    return result_str_lst

def main():
    print("Using features: ")
    print(*features_used, sep = ", ")
    print()

    print("Using time range: ")
    print(*features_time_range, sep = ", ")
    print()
    
    start_time = time.time()
    x,y = prepareTrainingData()
    classifier = train(x, y)
    end_time = time.time()
    elapsed_time = end_time -  start_time
    print()
    print('Trainingsdauer: ' + "{0:.2f}".format(elapsed_time) + ' s', end='\n')
    print()

    # gridSearch(x, y)

    result_str_lst = classify_2018(classifier)

    # Transfer list to DataFrame and save
    kaggle = pd.DataFrame(data=result_str_lst, columns=['Id', 'Category'])
    kaggle.shape

    kaggle = kaggle.to_csv('kaggle_Rapp_Katja_SVM_Grid_Search_test.csv', index=False)
    print()
    print("Done writing output file.")

if __name__ == "__main__":
    main()