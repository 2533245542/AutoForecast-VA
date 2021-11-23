import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class FilterController():
    def __init__(self, filter_class_to_kwargs_dictionary, lazy=False):
        '''
        input:
        filter_class_to_kwargs_dictionary
        lazy

        output:
        list_of_completed_filters
        filter_name_to_list_of_remained_clinics_dictionary
        filter_name_to_list_of_pruned_clinics_dictionary
        synthesized_list_of_remained_clinics   # the complementary set of pruned clinics
        synthesized_list_of_pruned_clinics

        steps:
        for each filter
            pass the respective kwargs to dataset
            get the list of remained and pruned list of clinics

        synthesized list of pruned clinics = combine all lists of pruned clinics
        synthesized list of remained clinics = get the complmentary set of syntheized list of pruned clincis

        '''
        # inputs
        self.filter_class_to_kwargs_dictionary = filter_class_to_kwargs_dictionary
        self.lazy = lazy

        # outputs
        self.filter_name_to_filter_dictionary = {}
        self.filter_name_to_list_of_remained_clinics_dictionary = {}
        self.filter_name_to_list_of_pruned_clinics_dictionary = {}
        self.synthesized_list_of_remained_clinics = []  # the intersection
        self.synthesized_list_of_pruned_clinics = []  # the union

        if not lazy:
            self.execute_filters()
            self.synthesize_results()

    def execute_filters(self):
        for filter_class in self.filter_class_to_kwargs_dictionary.keys():
            completed_filter = filter_class(**self.filter_class_to_kwargs_dictionary[filter_class])
            self.filter_name_to_filter_dictionary[filter_class.__name__] = completed_filter
            self.filter_name_to_list_of_remained_clinics_dictionary[filter_class.__name__] = completed_filter.list_of_remained_clinic
            self.filter_name_to_list_of_pruned_clinics_dictionary[filter_class.__name__] = completed_filter.list_of_pruned_clinic

    def synthesize_results(self):
        ''' For remained clinics, do intersection; forpruned clnics, do union; convert back to list in the end
        HACK: Covert each list to a set, then pass them all together to set.intersection
        '''
        self.synthesized_list_of_remained_clinics = set.intersection(*[set(list_of_remained_clinics) for list_of_remained_clinics in self.filter_name_to_list_of_remained_clinics_dictionary.values()])
        self.synthesized_list_of_pruned_clinics = set.union(*[set(list_of_pruned_clinics) for list_of_pruned_clinics in self.filter_name_to_list_of_pruned_clinics_dictionary.values()])

        self.synthesized_list_of_remained_clinics = list(self.synthesized_list_of_remained_clinics)
        self.synthesized_list_of_pruned_clinics = list(self.synthesized_list_of_pruned_clinics)

    def plot_a_clinic(self, dataset, clinic_name):
        '''input: dataset with continous days; the first column is named clinic, the second is named case
        Usage:
            figure = plot_a_clinic(dataset, '757')
            figure.show()
        '''

        figure, axis = plt.subplots(1)
        axis.plot(dataset[dataset.clinic == clinic_name].case)
        figure.suptitle('Case vs. time for clinic: ' + clinic_name)
        return figure


# filter_class_to_kwargs_dictionary = {FilterController: {'first_arg':1}}
# for filter in filter_class_to_kwargs_dictionary.keys():
#     filter(**filter_class_to_kwargs_dictionary[filter])
