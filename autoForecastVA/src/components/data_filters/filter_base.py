class FilterBase():
    def __init__(self):
        self.list_of_remained_clinic = []
        self.list_of_pruned_clinic = []

    def filter_clinic_by_maximally_allowable_value(self, clinic_to_value_dictionary, maximally_allowable_value, verbose=False):
        '''input: clinic_to_value_dictionary
        output: a list of remained clinics, a list of pruned clinics
        '''
        list_of_pruned_clinic = []
        list_of_remained_clinic = []
        for clinic in clinic_to_value_dictionary.keys():
            if clinic_to_value_dictionary[clinic] > maximally_allowable_value:
                list_of_pruned_clinic.append(clinic)
            else:
                list_of_remained_clinic.append(clinic)
        if verbose:
            self.verbose_printer(list_of_remained_clinic=list_of_remained_clinic, list_of_pruned_clinic=list_of_pruned_clinic)
        return list_of_remained_clinic, list_of_pruned_clinic

    def filter_clinic_by_minimally_allowable_value(self, clinic_to_value_dictionary, minimally_allowable_value,
                                                   verbose=False):
        '''input: clinic_to_value_dictionary
        output: a list of remained clinics, a list of pruned clinics
        '''
        list_of_pruned_clinic = []
        list_of_remained_clinic = []
        for clinic in clinic_to_value_dictionary.keys():
            if clinic_to_value_dictionary[clinic] < minimally_allowable_value:
                list_of_pruned_clinic.append(clinic)
            else:
                list_of_remained_clinic.append(clinic)
            if verbose:
                self.verbose_printer(list_of_remained_clinic=list_of_remained_clinic, list_of_pruned_clinic=list_of_pruned_clinic)
        return list_of_remained_clinic, list_of_pruned_clinic

    def verbose_printer(self, list_of_remained_clinic, list_of_pruned_clinic):
        total_number_of_clinics = len(list_of_remained_clinic) + len(list_of_pruned_clinic)
        number_of_remained_clinics = len(list_of_remained_clinic)
        number_of_pruned_clinics = len(list_of_pruned_clinic)
        percentage_of_remained_clinics = round(len(list_of_remained_clinic) / total_number_of_clinics, 3) * 100  # in 3 digits
        percentage_of_pruned_clinics = round(len(list_of_pruned_clinic) / total_number_of_clinics, 4) * 100
        print('{}:'.format(self.__class__.__name__), 'Number of clinics remained {} ({}%); pruned {} ({}%):, '.format(number_of_remained_clinics, percentage_of_remained_clinics, number_of_pruned_clinics, percentage_of_pruned_clinics))
