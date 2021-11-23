import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import pickle
'''
data_clusetering_and_normalization.py
'''
class DataCluseteringAndDoubleNormalization():
    def __init__(self, dataDelimitingByDayAndMedicalCenter, max_number_of_cluster=10, number_of_test_day_in_day_level_DataFrame=1, lazy=False, use_cache=False, save_as_cache=False, cache_path='../../../data/cache/data_clustering_and_normalization_cache'):
        self.max_number_of_cluster = max_number_of_cluster
        self.number_of_test_day_in_day_level_DataFrame = number_of_test_day_in_day_level_DataFrame
        self.cache_path = cache_path
        self.medical_center_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_medical_center_level_DataFrame_dictionary()
        self.medical_center_level_day_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_medical_center_level_day_level_DataFrame_dictionary()
        self.day_level_medical_center_level_DataFrame_dictionary = dataDelimitingByDayAndMedicalCenter.get_day_level_medical_center_level_DataFrame_dictionary()
        self.day_level_medical_center_to_cluster_dictionary = {}
        self.day_level_cluster_level_medical_center_list_dictionary = {}
        self.day_level_medical_center_level_normalization_statistics_dictionary = {}
        self.day_level_medical_center_level_within_medical_center_normalized_DataFrame_dictionary = {}
        self.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary = {}
        self.day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary = {}
        self.day_level_cluster_level_combined_normalization_statistics_dictionary = {}
        self.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary = {}
        self.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary = {}
        self.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary = {}

        if use_cache:
            # load from the cache file path
            with open(cache_path, "rb") as cache:
                self.day_level_medical_center_to_cluster_dictionary = pickle.load(cache)
                self.day_level_cluster_level_medical_center_list_dictionary = pickle.load(cache)
                self.day_level_medical_center_level_normalization_statistics_dictionary = pickle.load(cache)
                self.day_level_medical_center_level_within_medical_center_normalized_DataFrame_dictionary = pickle.load(cache)
                self.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary = pickle.load(cache)
                self.day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary = pickle.load(cache)
                self.day_level_cluster_level_combined_normalization_statistics_dictionary = pickle.load(cache)
                self.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary = pickle.load(cache)
                self.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary = pickle.load(cache)
                self.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary = pickle.load(cache)
        else:
            if not lazy:
                self.clustering()
                self.within_medical_center_normalization()
                self.across_medical_center_normalization()
                self.split_and_combine_data_back_to_day_level_DataFrame()
                if save_as_cache:
                    self.save_result_as_cache()


    def clustering(self):
        ''' Find the cluster membership for each medical center and each day'''
        for day in self.day_level_medical_center_level_DataFrame_dictionary.keys():
            list_of_medical_center = []
            list_of_case_series = []

            for medical_center in self.day_level_medical_center_level_DataFrame_dictionary[day].keys():
                day_level_DataFrame = self.day_level_medical_center_level_DataFrame_dictionary[day][medical_center]
                list_of_medical_center.append(medical_center)
                list_of_case_series.append(day_level_DataFrame.case)
            X = pd.concat(list_of_case_series, axis=1).transpose().to_numpy()
            best_calinski_harabasz_score = np.NINF  # the larger the better
            best_cluster_algorithm = None
            best_number_of_cluster = 0
            for current_number_of_cluster in range(2, self.max_number_of_cluster + 1):  # need to be minimum of two clusters
                current_cluster_algorithm = AgglomerativeClustering(n_clusters=current_number_of_cluster, linkage='complete')
                current_cluster_algorithm.fit_predict(X)
                current_calinski_harabasz_score = metrics.calinski_harabasz_score(X, current_cluster_algorithm.labels_)
                if current_calinski_harabasz_score > best_calinski_harabasz_score:
                    best_calinski_harabasz_score = current_calinski_harabasz_score
                    best_cluster_algorithm = current_cluster_algorithm
                    best_number_of_cluster = current_number_of_cluster

            # create mappings between clusters and medical centers; create mappings between a cluster and a list of its medical centers
            cluster_level_medical_center_dictionary = {}
            for i in range(best_number_of_cluster):
                cluster_level_medical_center_dictionary[i] = []

            medical_center_to_cluster_dictionary = {}
            for cluster, medical_center in zip(best_cluster_algorithm.labels_, list_of_medical_center):
                medical_center_to_cluster_dictionary[medical_center] = cluster
                cluster_level_medical_center_dictionary[cluster].append(medical_center)

            self.day_level_medical_center_to_cluster_dictionary[day] = medical_center_to_cluster_dictionary
            self.day_level_cluster_level_medical_center_list_dictionary[day] = cluster_level_medical_center_dictionary

    def within_medical_center_normalization(self):
        '''Perform normalization for each medical center on each day'''
        for day in self.day_level_medical_center_level_DataFrame_dictionary.keys():
            medical_center_level_normalization_statistics_dictionary = {}
            medical_center_level_normalized_DataFrame_dictionary = {}
            for medical_center in self.medical_center_level_day_level_DataFrame_dictionary.keys():
                day_level_DataFrame = self.day_level_medical_center_level_DataFrame_dictionary[day][medical_center]
                mean = day_level_DataFrame.iloc[:-self.number_of_test_day_in_day_level_DataFrame, 1:].mean()
                std = day_level_DataFrame.iloc[:-self.number_of_test_day_in_day_level_DataFrame, 1:].std().replace(0,1)
                normalized_day_level_DataFrame = (day_level_DataFrame.iloc[:, 1:] - mean)/std  # normalization based on the training observations; also a hack to avoid dividing by zero; also the standard deviation is calculated using sample standard deviation
                normalized_day_level_DataFrame.insert(0, 'clinic', medical_center)
                medical_center_level_normalization_statistics_dictionary[medical_center] = (mean, std)
                medical_center_level_normalized_DataFrame_dictionary[medical_center] = normalized_day_level_DataFrame
            self.day_level_medical_center_level_normalization_statistics_dictionary[day] = medical_center_level_normalization_statistics_dictionary
            self.day_level_medical_center_level_within_medical_center_normalized_DataFrame_dictionary[day] = medical_center_level_normalized_DataFrame_dictionary


    def across_medical_center_normalization(self):
        '''Perform normalization for each cluster on each day'''
        # for each day and for each cluster and for each medical center corresponding to the cluster; split DataFrames to training and testing; combine the centers to training and testing; calculate statistics on the training data; normalize on both training and testing; record data

        ''' 
        We normalize in a cluster across medical centers based on the training data. Cluster 0 contains 111 and 222. The training data are {'case': [1,2,1,2], 'call': [1,2,1,2]}. The mean is [1.5,1.5] and the std is [0.577,0.577]. Cluster 1 contains 333. The training data are {'case': [1,1], 'call': [1,2]}. The mean is [1,1.5] and the std is [1,0.707106]
        '''

        for day in self.day_level_medical_center_level_DataFrame_dictionary.keys():
            cluster_level_not_normalized_combined_train_DataFrame_dictionary = {}
            cluster_level_not_normalized_combined_test_DataFrame_dictionary = {}
            cluster_level_combined_normalization_statistics_dictionary = {}
            cluster_level_normalized_combined_train_DataFrame_dictionary = {}
            cluster_level_normalized_combined_test_DataFrame_dictionary = {}
            for cluster in self.day_level_cluster_level_medical_center_list_dictionary[day].keys():
                train_day_level_DataFrame_list = []
                test_day_level_DataFrame_list = []
                for medical_center in self.day_level_cluster_level_medical_center_list_dictionary[day][cluster]:
                    train_day_level_DataFrame_list.append(self.day_level_medical_center_level_within_medical_center_normalized_DataFrame_dictionary[day][medical_center].iloc[:-self.number_of_test_day_in_day_level_DataFrame, :])
                    test_day_level_DataFrame_list.append(self.day_level_medical_center_level_within_medical_center_normalized_DataFrame_dictionary[day][medical_center].iloc[-self.number_of_test_day_in_day_level_DataFrame:, :])
                combined_train_DataFrame = pd.concat(train_day_level_DataFrame_list)
                combined_test_DataFrame = pd.concat(test_day_level_DataFrame_list)
                cluster_level_not_normalized_combined_train_DataFrame_dictionary[cluster] = combined_train_DataFrame
                cluster_level_not_normalized_combined_test_DataFrame_dictionary[cluster] = combined_test_DataFrame
                mean = combined_train_DataFrame.iloc[:,1:].mean()
                std = combined_train_DataFrame.iloc[:,1:].std().replace(0,1)   # same trick as before
                normalized_combined_train_DataFrame = (combined_train_DataFrame.iloc[:,1:] - mean) / std
                normalized_combined_test_DataFrame = (combined_test_DataFrame.iloc[:,1:] - mean) / std
                normalized_combined_train_DataFrame.insert(0, 'clinic', combined_train_DataFrame.clinic)
                normalized_combined_test_DataFrame.insert(0, 'clinic', combined_test_DataFrame.clinic)
                cluster_level_combined_normalization_statistics_dictionary[cluster] = (mean, std)
                cluster_level_normalized_combined_train_DataFrame_dictionary[cluster] = normalized_combined_train_DataFrame
                cluster_level_normalized_combined_test_DataFrame_dictionary[cluster] = normalized_combined_test_DataFrame

            self.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary[day] = cluster_level_not_normalized_combined_train_DataFrame_dictionary
            self.day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary[day] = cluster_level_not_normalized_combined_test_DataFrame_dictionary
            self.day_level_cluster_level_combined_normalization_statistics_dictionary[day] = cluster_level_combined_normalization_statistics_dictionary
            self.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary[day] = cluster_level_normalized_combined_train_DataFrame_dictionary
            self.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary[day] = cluster_level_normalized_combined_test_DataFrame_dictionary

    def split_and_combine_data_back_to_day_level_DataFrame(self):
        # for each day and medical center, combine the normalized train and test DataFrame back to a day-level DataFrame
        for day in self.day_level_medical_center_level_DataFrame_dictionary.keys():
            medical_center_level_across_medical_center_normalized_DataFrame_dictionary = {}
            for cluster in self.day_level_cluster_level_medical_center_list_dictionary[day].keys():
                for medical_center in self.day_level_cluster_level_medical_center_list_dictionary[day][cluster]:
                    normalized_combined_train_DataFrame = self.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary[day][cluster]
                    normalized_combined_test_DataFrame = self.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary[day][cluster]
                    dataFrame = pd.concat([normalized_combined_train_DataFrame[normalized_combined_train_DataFrame.clinic == medical_center], normalized_combined_test_DataFrame[normalized_combined_test_DataFrame.clinic == medical_center]])
                    medical_center_level_across_medical_center_normalized_DataFrame_dictionary[medical_center] = dataFrame
            self.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary[day] = medical_center_level_across_medical_center_normalized_DataFrame_dictionary
    def save_result_as_cache(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.day_level_medical_center_to_cluster_dictionary, f)
            pickle.dump(self.day_level_cluster_level_medical_center_list_dictionary, f)
            pickle.dump(self.day_level_medical_center_level_normalization_statistics_dictionary, f)
            pickle.dump(self.day_level_medical_center_level_within_medical_center_normalized_DataFrame_dictionary, f)
            pickle.dump(self.day_level_cluster_level_not_normalized_combined_train_DataFrame_dictionary, f)
            pickle.dump(self.day_level_cluster_level_not_normalized_combined_test_DataFrame_dictionary, f)
            pickle.dump(self.day_level_cluster_level_combined_normalization_statistics_dictionary, f)
            pickle.dump(self.day_level_cluster_level_normalized_combined_train_DataFrame_dictionary, f)
            pickle.dump(self.day_level_cluster_level_normalized_combined_test_DataFrame_dictionary, f)
            pickle.dump(self.day_level_medical_center_level_across_medical_center_normalized_DataFrame_dictionary, f)





