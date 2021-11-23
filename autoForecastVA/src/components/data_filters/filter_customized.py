from autoForecastVA.src.components.data_filters.filter_base import FilterBase
class FilterProportionOfMissingDays(FilterBase):
    def __init__(self, dataset, maximum_allowable_proportion_of_missing_data=1, verbose=False, lazy=False):
        super().__init__()
        # inputs
        self.dataset = dataset
        self.maximum_allowable_proportion_of_missing_data = maximum_allowable_proportion_of_missing_data
        self.verbose = verbose

        # outputs
        self.clinic_to_number_of_total_days_dictionary = {}
        self.clinic_to_number_of_missing_days_dictionary = {}
        self.clinic_to_proportion_of_missing_days_dictionary = {}
        self.list_of_remained_clinic = []
        self.list_of_pruned_clinic = []

        if not lazy:
            self.gather_clinic_level_statistics()
            self.list_of_remained_clinic, self.list_of_pruned_clinic = self.filter_clinic_by_maximally_allowable_value(clinic_to_value_dictionary=self.clinic_to_proportion_of_missing_days_dictionary, maximally_allowable_value=self.maximum_allowable_proportion_of_missing_data, verbose=self.verbose)

    def gather_clinic_level_statistics(self):
        for clinic in self.dataset.clinic.unique():
            number_of_total_days = len(self.dataset[self.dataset.clinic == clinic].case)
            number_of_missing_days = self.dataset[self.dataset.clinic == clinic].case.isna().sum()
            proportion_of_missing_days = number_of_missing_days / number_of_total_days

            self.clinic_to_number_of_total_days_dictionary[clinic] = number_of_total_days
            self.clinic_to_number_of_missing_days_dictionary[clinic] = number_of_missing_days
            self.clinic_to_proportion_of_missing_days_dictionary[clinic] = proportion_of_missing_days

class FilterStandardDeviationDividedByMean(FilterBase):
    def __init__(self, dataset, maximum_allowable_divided_value=1, verbose=False, lazy=False):
        super().__init__()
        # inputs
        self.dataset = dataset
        self.maximum_allowable_divided_value = maximum_allowable_divided_value
        self.verbose = verbose
        self.lazy = lazy

        # outputs
        self.clinic_to_std_dictionary = {}  # first remove NA/null, then calculate
        self.clinic_to_mean_dictionary = {}  # first remove NA/null, then calculate
        self.clinic_to_std_divided_by_mean_dictionary = {}
        self.list_of_remained_clinic = []
        self.list_of_pruned_clinic = []

        if not lazy:
            self.gather_clinic_level_statistics()
            self.list_of_remained_clinic, self.list_of_pruned_clinic = self.filter_clinic_by_maximally_allowable_value(clinic_to_value_dictionary=self.clinic_to_std_divided_by_mean_dictionary, maximally_allowable_value=self.maximum_allowable_divided_value, verbose=self.verbose)


    def gather_clinic_level_statistics(self):
        for clinic in self.dataset.clinic.unique():
            std = self.dataset[self.dataset.clinic == clinic].case.std()
            mean = self.dataset[self.dataset.clinic == clinic].case.mean()

            self.clinic_to_std_dictionary[clinic] = std
            self.clinic_to_mean_dictionary[clinic] = mean
            self.clinic_to_std_divided_by_mean_dictionary[clinic] = std / mean
