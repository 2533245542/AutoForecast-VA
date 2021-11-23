import copy

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import pickle

from autoForecastVA.src.utils.tuning_interval_moderator import OperationIntervalModerator

'''
Only across medical center normalization is done.
'''
class DataCluseteringAndNormalization():
    def __init__(self, dataDelimitingByDayAndMedicalCenter, max_number_of_cluster=10, number_of_test_day_in_day_level_DataFrame=1, operation_interval=1, lazy=False, use_cache=False, save_as_cache=False, do_clustering=True, do_normalization=True, cache_path='../../../data/cache/data_clustering_and_normalization_cache'):
        # inputs
        self.max_number_of_cluster = max_number_of_cluster
        self.number_of_test_day_in_day_level_DataFrame = number_of_test_day_in_day_level_DataFrame
        self.operation_interval = operation_interval
        self.lazy = lazy
        self.use_cache = use_cache
        self.save_as_cache = save_as_cache
        self.do_clustering = do_clustering
        self.do_normalization = do_normalization
        self.cache_path = cache_path
        self.medical_center_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_medical_center_level_DataFrame_dictionary()
        self.medical_center_level_day_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_medical_center_level_day_level_DataFrame_dictionary()
        self.day_level_medical_center_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_day_level_medical_center_level_DataFrame_dictionary()

        # outputs
        self.day_level_medical_center_to_cluster_dictionary = {}
        self.day_level_cluster_level_medical_center_list_dictionary = {}
        self.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary = {}
        self.day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary = {}
        self.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary = {}
        self.day_level_cluster_level_combined_normalization_statistics_dictionary = {}  # a value is in the form of (mean, std)
        self.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary = {}
        self.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary = {}
        self.day_level_cluster_level_normalized_combined_DataFrame_dictionary = {}
        self.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary = {}
        self.day_level_cluster_level_medical_center_level_DataFrame_dictionary = {}
        self.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary = {}

        if use_cache:
            # load from the cache file path
            with open(cache_path, "rb") as cache:
                self.day_level_medical_center_to_cluster_dictionary = pickle.load(cache)
                self.day_level_cluster_level_medical_center_list_dictionary = pickle.load(cache)
                self.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary = pickle.load(cache)
                self.day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary = pickle.load(cache)
                self.day_level_cluster_level_combined_normalization_statistics_dictionary = pickle.load(cache)
                self.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary = pickle.load(cache)
                self.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary = pickle.load(cache)
                self.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary = pickle.load(cache)
        else:
            if not lazy:
                self.clustering()
                self.across_medical_center_normalization()
                self.split_and_combine_data_back_to_day_level_DataFrame()
                if save_as_cache:
                    self.save_result_as_cache()


    def clustering(self):
        ''' Find the cluster membership for each medical center and each day'''
        for day in self.day_level_medical_center_level_DataFrame_dictionary.keys():
            list_of_medical_center = []
            list_of_case_series = []

            # get the case feature of all medical centers for this day
            for medical_center in self.day_level_medical_center_level_DataFrame_dictionary[day].keys():
                day_level_DataFrame = self.day_level_medical_center_level_DataFrame_dictionary[day][medical_center]
                list_of_medical_center.append(medical_center)
                list_of_case_series.append(day_level_DataFrame.case)
            X = pd.concat(list_of_case_series, axis=1).transpose().to_numpy()  # each series matches a medical center
            best_calinski_harabasz_score = np.NINF  # the larger the better
            best_cluster_algorithm = None
            best_number_of_cluster = 0  # optimal number of clusters

            real_min_number_of_cluster = 2  # need to be minimum of two clusters (inclusive)
            real_max_number_of_cluster = min(self.max_number_of_cluster, X.shape[0] - 1)  # maximum of (number of features - 1) inclusive

            for current_number_of_cluster in range(real_min_number_of_cluster, real_max_number_of_cluster + 1):
                current_cluster_algorithm = AgglomerativeClustering(n_clusters=current_number_of_cluster, linkage='complete')
                current_cluster_algorithm.fit_predict(X)
                current_calinski_harabasz_score = metrics.calinski_harabasz_score(X, current_cluster_algorithm.labels_)
                if current_calinski_harabasz_score > best_calinski_harabasz_score:
                    best_calinski_harabasz_score = current_calinski_harabasz_score
                    best_cluster_algorithm = current_cluster_algorithm
                    best_number_of_cluster = current_number_of_cluster

            # create mappings between a medical center and a cluster; create mappings between a cluster and a list of medical centers
            cluster_to_medical_center_list_dictionary = {}

            if not self.do_clustering:
                best_number_of_cluster = len(list_of_medical_center)

            for i in range(best_number_of_cluster):
                cluster_to_medical_center_list_dictionary[i] = []

            medical_center_to_cluster_dictionary = {}

            if not self.do_clustering:
                best_cluster_algorithm.labels_ = list(range(len(list_of_medical_center)))

            for cluster, medical_center in zip(best_cluster_algorithm.labels_, list_of_medical_center):  # each series matches a medical center
                medical_center_to_cluster_dictionary[medical_center] = cluster
                cluster_to_medical_center_list_dictionary[cluster].append(medical_center)

            self.day_level_medical_center_to_cluster_dictionary[day] = medical_center_to_cluster_dictionary
            self.day_level_cluster_level_medical_center_list_dictionary[day] = cluster_to_medical_center_list_dictionary

            operationIntervalModerator = OperationIntervalModerator(days=self.day_level_medical_center_level_DataFrame_dictionary.keys(), operation_interval=self.operation_interval)
            current_operation_day = operationIntervalModerator.get_operation_day_of_a_day(day=day)
            self.day_level_medical_center_to_cluster_dictionary[day] = copy.deepcopy(self.day_level_medical_center_to_cluster_dictionary[current_operation_day])
            self.day_level_cluster_level_medical_center_list_dictionary[day] = copy.deepcopy(self.day_level_cluster_level_medical_center_list_dictionary[current_operation_day])

    def across_medical_center_normalization(self):
        '''Perform normalization for each cluster on each day'''
        # for each day and for each cluster and for each medical center corresponding to the cluster; split DataFrames to training and testing; combine the centers to training and testing; calculate statistics on the training data; normalize on both training and testing; record data

        ''' 
        We normalize in a cluster across medical centers based on the training data. Cluster 0 contains 111 and 222. The training data are {'case': [1,2,1,2], 'call': [1,2,1,2]}. The mean is [1.5,1.5] and the std is [0.577,0.577]. Cluster 1 contains 333. The training data are {'case': [1,1], 'call': [1,2]}. The mean is [1,1.5] and the std is [1,0.707106]
        '''

        for day in self.day_level_cluster_level_medical_center_list_dictionary.keys():
            cluster_level_not_normalized_combined_train_DataFrame_dictionary = {}
            cluster_level_not_normalized_combined_test_DataFrame_dictionary = {}
            cluster_level_not_normalized_combined_DataFrame_dictionary = {}  # new
            cluster_level_combined_normalization_statistics_dictionary = {}
            cluster_level_normalized_combined_train_DataFrame_dictionary = {}
            cluster_level_normalized_combined_test_DataFrame_dictionary = {}
            cluster_level_normalized_combined_DataFrame_dictionary = {}  # new
            for cluster in self.day_level_cluster_level_medical_center_list_dictionary[day].keys():
                train_day_level_DataFrame_list = []
                test_day_level_DataFrame_list = []
                for medical_center in self.day_level_cluster_level_medical_center_list_dictionary[day][cluster]:
                    train_day_level_DataFrame_list.append(self.day_level_medical_center_level_DataFrame_dictionary[day][medical_center].iloc[:-self.number_of_test_day_in_day_level_DataFrame, :])
                    test_day_level_DataFrame_list.append(self.day_level_medical_center_level_DataFrame_dictionary[day][medical_center].iloc[-self.number_of_test_day_in_day_level_DataFrame:, :])
                combined_train_DataFrame = pd.concat(train_day_level_DataFrame_list)
                combined_test_DataFrame = pd.concat(test_day_level_DataFrame_list)
                combined_DataFrame = pd.concat([combined_train_DataFrame, combined_test_DataFrame])
                cluster_level_not_normalized_combined_train_DataFrame_dictionary[cluster] = combined_train_DataFrame
                cluster_level_not_normalized_combined_test_DataFrame_dictionary[cluster] = combined_test_DataFrame
                cluster_level_not_normalized_combined_DataFrame_dictionary[cluster] = combined_DataFrame

                mean = combined_train_DataFrame.iloc[:,1:].mean()
                std = combined_train_DataFrame.iloc[:,1:].std().replace(0,1)   # same trick as before
                if not self.do_normalization:  # set mean to 0 and std to 1 so the below calculatation will have no effect
                    mean.iloc[:] = 0
                    std.iloc[:] = 1
                normalized_combined_train_DataFrame = (combined_train_DataFrame.iloc[:,1:] - mean) / std
                normalized_combined_test_DataFrame = (combined_test_DataFrame.iloc[:,1:] - mean) / std
                normalized_combined_train_DataFrame.insert(0, 'clinic', combined_train_DataFrame.clinic)
                normalized_combined_test_DataFrame.insert(0, 'clinic', combined_test_DataFrame.clinic)
                normalized_combined_DataFrame = pd.concat([normalized_combined_train_DataFrame, normalized_combined_test_DataFrame])
                cluster_level_combined_normalization_statistics_dictionary[cluster] = (mean, std)

                cluster_level_normalized_combined_train_DataFrame_dictionary[cluster] = normalized_combined_train_DataFrame
                cluster_level_normalized_combined_test_DataFrame_dictionary[cluster] = normalized_combined_test_DataFrame
                cluster_level_normalized_combined_DataFrame_dictionary[cluster] = normalized_combined_DataFrame

            self.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[day] = cluster_level_not_normalized_combined_train_DataFrame_dictionary
            self.day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary[day] = cluster_level_not_normalized_combined_test_DataFrame_dictionary
            self.day_level_cluster_level_not_normalized_combined_DataFrame_dictionary[day] = cluster_level_not_normalized_combined_DataFrame_dictionary  # new
            self.day_level_cluster_level_combined_normalization_statistics_dictionary[day] = cluster_level_combined_normalization_statistics_dictionary
            self.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary[day] = cluster_level_normalized_combined_train_DataFrame_dictionary
            self.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary[day] = cluster_level_normalized_combined_test_DataFrame_dictionary
            self.day_level_cluster_level_normalized_combined_DataFrame_dictionary[day] = cluster_level_normalized_combined_DataFrame_dictionary  # new

    def split_and_combine_data_back_to_day_level_DataFrame(self):
        # for each day and medical center, combine the normalized train and test DataFrame back to a day-level DataFrame
        for day in self.day_level_cluster_level_medical_center_list_dictionary.keys():
            medical_center_level_across_medical_center_normalized_DataFrame_dictionary = {}
            for cluster in self.day_level_cluster_level_medical_center_list_dictionary[day].keys():
                for medical_center in self.day_level_cluster_level_medical_center_list_dictionary[day][cluster]:
                    normalized_combined_train_DataFrame = self.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary[day][cluster]
                    normalized_combined_test_DataFrame = self.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary[day][cluster]
                    dataFrame = pd.concat([normalized_combined_train_DataFrame[normalized_combined_train_DataFrame.clinic == medical_center], normalized_combined_test_DataFrame[normalized_combined_test_DataFrame.clinic == medical_center]])
                    medical_center_level_across_medical_center_normalized_DataFrame_dictionary[medical_center] = dataFrame
            self.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day] = medical_center_level_across_medical_center_normalized_DataFrame_dictionary



        # create a day-level, cluster-level, medical-center level original DataFrame dictionary for the convienece of future parts
        for day in self.day_level_cluster_level_medical_center_list_dictionary.keys():
            cluster_level_medical_center_level_DataFrame_dictionary = {}
            for cluster in self.day_level_cluster_level_medical_center_list_dictionary[day].keys():
                medical_center_level_DataFrame_dictionary = {}
                for medical_center in self.day_level_cluster_level_medical_center_list_dictionary[day][cluster]:
                    medical_center_level_DataFrame_dictionary[medical_center] = self.day_level_medical_center_level_DataFrame_dictionary[day][medical_center]
                cluster_level_medical_center_level_DataFrame_dictionary[cluster] = medical_center_level_DataFrame_dictionary
            self.day_level_cluster_level_medical_center_level_DataFrame_dictionary[day] = cluster_level_medical_center_level_DataFrame_dictionary


        # create a day-level, cluster-level, medical-center level normalized DataFrame dictionary for the convienece of future parts
        for day in self.day_level_medical_center_level_DataFrame_dictionary.keys():
            cluster_level_medical_center_level_DataFrame_dictionary = {}
            for cluster in self.day_level_cluster_level_medical_center_list_dictionary[day].keys():
                medical_center_level_DataFrame_dictionary = {}
                for medical_center in self.day_level_cluster_level_medical_center_list_dictionary[day][cluster]:
                    medical_center_level_DataFrame_dictionary[medical_center] = self.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day][medical_center]
                cluster_level_medical_center_level_DataFrame_dictionary[cluster] = medical_center_level_DataFrame_dictionary
            self.day_level_cluster_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day] = cluster_level_medical_center_level_DataFrame_dictionary


    def save_result_as_cache(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.day_level_medical_center_to_cluster_dictionary, f)
            pickle.dump(self.day_level_cluster_level_medical_center_list_dictionary, f)
            pickle.dump(self.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary, f)
            pickle.dump(self.day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary, f)
            pickle.dump(self.day_level_cluster_level_combined_normalization_statistics_dictionary, f)
            pickle.dump(self.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary, f)
            pickle.dump(self.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary, f)
            pickle.dump(self.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary, f)





