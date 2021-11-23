from autoForecastVA.src.components.data_filters.filter_customized import FilterProportionOfMissingDays, FilterStandardDeviationDividedByMean
from autoForecastVA.src.components.data_preprocessing_global.data_imputation_and_averaging import DataImputationAndAveraging
from autoForecastVA.src.components.data_filters.filter_controller import FilterController

import unittest

class TestFilterController(unittest.TestCase):
    def test_toy_example(self):
        dataset_path = '../../../data/coviddata07292020.csv'
        dataImputationAndAveraging = DataImputationAndAveraging(dataset_path=dataset_path, lazy=True, number_of_days_for_data_averaging=0)
        dataImputationAndAveraging.read_dataset()
        dataset = dataImputationAndAveraging.get_processed_dataset()

        maximum_allowable_proportion_of_missing_data = 0.05
        maximum_allowable_divided_value = 1
        filter_class_to_kwargs_dictionary = {FilterProportionOfMissingDays: {'dataset':dataset, 'maximum_allowable_proportion_of_missing_data': maximum_allowable_proportion_of_missing_data, 'verbose': True}, FilterStandardDeviationDividedByMean: {'dataset':dataset, 'maximum_allowable_divided_value':maximum_allowable_divided_value, 'verbose':True}}
        filterController = FilterController(filter_class_to_kwargs_dictionary=filter_class_to_kwargs_dictionary)

        self.assertListEqual(filterController.filter_name_to_filter_dictionary['FilterProportionOfMissingDays'].list_of_remained_clinic, ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501', '503', '504', '506', '508', '509', '512', '515', '516', '517', '518', '519', '520', '521', '523', '526', '528', '528A6', '528A7', '528A8', '529', '531', '534', '537', '539', '541', '542', '544', '546', '548', '549', '550', '552', '553', '554', '556', '557', '558', '561', '562', '564', '565', '568', '570', '573', '575', '578', '580', '581', '583', '585', '586', '589', '589A4', '589A5', '589A7', '590', '593', '595', '596', '598', '600', '603', '605', '607', '608', '610', '612A4', '613', '614', '618', '619', '620', '621', '623', '626', '630', '631', '632', '635', '636', '636A6', '636A8', '637', '640', '642', '644', '646', '648', '649', '650', '652', '654', '655', '656', '657', '657A5', '658', '659', '660', '662', '663', '664', '668', '671', '673', '674', '675', '676', '678', '688', '689', '691', '693', '695', '740', '756', '757'])
        self.assertListEqual(filterController.filter_name_to_filter_dictionary['FilterProportionOfMissingDays'].list_of_pruned_clinic, ['358', '502', '538', '540', '629', '653', '657A4', '666', '667', '672', '679', '687', '692'])

        self.assertListEqual(filterController.filter_name_to_filter_dictionary['FilterStandardDeviationDividedByMean'].list_of_remained_clinic[:10], ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501'])
        self.assertListEqual(filterController.filter_name_to_filter_dictionary['FilterStandardDeviationDividedByMean'].list_of_pruned_clinic[:5], ['358', '502', '504', '517', '520'])

        self.assertEqual(len(set(filterController.synthesized_list_of_remained_clinics).intersection(filterController.synthesized_list_of_pruned_clinics)), 0)
        self.assertEqual(len(set(filterController.synthesized_list_of_remained_clinics).union(filterController.synthesized_list_of_pruned_clinics)), 140)

        self.assertEqual(len(filterController.synthesized_list_of_remained_clinics), 110)
        self.assertEqual(len(filterController.synthesized_list_of_pruned_clinics), 30)

        # 110 remained, 30 pruned
        self.assertListEqual(sorted(filterController.synthesized_list_of_remained_clinics), ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501', '503', '506', '508', '509', '512', '515', '516', '518', '519', '521', '523', '528', '528A6', '528A7', '528A8', '529', '531', '534', '537', '539', '541', '542', '544', '548', '549', '550', '552', '554', '556', '557', '558', '562', '568', '570', '573', '575', '578', '580', '581', '583', '585', '589', '589A5', '589A7', '590', '593', '595', '596', '600', '603', '605', '607', '608', '610', '612A4', '614', '618', '619', '621', '623', '626', '631', '635', '636', '636A6', '636A8', '637', '640', '642', '644', '646', '648', '649', '650', '652', '654', '655', '656', '657', '657A5', '658', '659', '660', '662', '663', '664', '668', '671', '673', '674', '675', '676', '678', '688', '689', '691', '693', '695', '756', '757'])
        self.assertListEqual(sorted(filterController.synthesized_list_of_pruned_clinics), ['358', '502', '504', '517', '520', '526', '538', '540', '546', '553', '561', '564', '565', '586', '589A4', '598', '613', '620', '629', '630', '632', '653', '657A4', '666', '667', '672', '679', '687', '692', '740'])
        print('Synthesized remained clinics:', filterController.synthesized_list_of_remained_clinics)
        print('Synthesized pruned clinics:', filterController.synthesized_list_of_pruned_clinics)

        ## debug start
        filterController.filter_name_to_filter_dictionary['FilterStandardDeviationDividedByMean'].clinic_to_std_dictionary
        filterController.filter_name_to_filter_dictionary['FilterStandardDeviationDividedByMean'].clinic_to_mean_dictionary
        filterController.filter_name_to_filter_dictionary['FilterStandardDeviationDividedByMean'].clinic_to_std_divided_by_mean_dictionary
        filterController.filter_name_to_filter_dictionary['FilterStandardDeviationDividedByMean'].list_of_remained_clinic
        filterController.filter_name_to_filter_dictionary['FilterStandardDeviationDividedByMean'].list_of_pruned_clinic

        for clinic in filterController.filter_name_to_filter_dictionary['FilterStandardDeviationDividedByMean'].list_of_pruned_clinic:
            print(clinic)
            print(filterController.filter_name_to_filter_dictionary[ 'FilterStandardDeviationDividedByMean'].clinic_to_std_dictionary[clinic])
            print(filterController.filter_name_to_filter_dictionary[ 'FilterStandardDeviationDividedByMean'].clinic_to_mean_dictionary[clinic])
            print(filterController.filter_name_to_filter_dictionary[ 'FilterStandardDeviationDividedByMean'].clinic_to_std_divided_by_mean_dictionary[clinic])

        ## debug end