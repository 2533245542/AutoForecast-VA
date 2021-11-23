from autoForecastVA.src.components.data_filters.filter_customized import FilterProportionOfMissingDays, FilterStandardDeviationDividedByMean
from autoForecastVA.src.components.data_preprocessing_global.data_imputation_and_averaging import DataImputationAndAveraging
import unittest
class TestFilterCustomized(unittest.TestCase):
    def test_toy(self):
        ''' We call each filter with dataset and the respective arguments. We check for the resulted values and clinics '''
        dataset_path = '../../../data/coviddata07292020.csv'
        dataImputationAndAveraging = DataImputationAndAveraging(dataset_path=dataset_path, lazy=True, number_of_days_for_data_averaging=0)
        dataImputationAndAveraging.read_dataset()
        dataset = dataImputationAndAveraging.get_processed_dataset()
        self.assertEqual(len(dataset.clinic.unique()), 140)

        '''test missing porportion day filter'''
        maximum_allowable_proportion_of_missing_data = 0.05
        filterProportionOfMissingDays = FilterProportionOfMissingDays(dataset=dataset, maximum_allowable_proportion_of_missing_data=maximum_allowable_proportion_of_missing_data, verbose=True)
        self.assertListEqual(filterProportionOfMissingDays.list_of_remained_clinic, ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501', '503', '504', '506', '508', '509', '512', '515', '516', '517', '518', '519', '520', '521', '523', '526', '528', '528A6', '528A7', '528A8', '529', '531', '534', '537', '539', '541', '542', '544', '546', '548', '549', '550', '552', '553', '554', '556', '557', '558', '561', '562', '564', '565', '568', '570', '573', '575', '578', '580', '581', '583', '585', '586', '589', '589A4', '589A5', '589A7', '590', '593', '595', '596', '598', '600', '603', '605', '607', '608', '610', '612A4', '613', '614', '618', '619', '620', '621', '623', '626', '630', '631', '632', '635', '636', '636A6', '636A8', '637', '640', '642', '644', '646', '648', '649', '650', '652', '654', '655', '656', '657', '657A5', '658', '659', '660', '662', '663', '664', '668', '671', '673', '674', '675', '676', '678', '688', '689', '691', '693', '695', '740', '756', '757'])
        self.assertListEqual(filterProportionOfMissingDays.list_of_pruned_clinic, ['358', '502', '538', '540', '629', '653', '657A4', '666', '667', '672', '679', '687', '692'])

        '''test mean divided by standard deviation filter'''
        maximum_allowable_divided_value = 1
        filterStandardDeviationDividedByMean = FilterStandardDeviationDividedByMean(dataset=dataset, maximum_allowable_divided_value=maximum_allowable_divided_value, verbose=True)
        self.assertListEqual(filterStandardDeviationDividedByMean.list_of_remained_clinic[:10], ['402', '405', '436', '437', '438', '442', '459', '460', '463', '501'])
        self.assertListEqual(filterStandardDeviationDividedByMean.list_of_pruned_clinic[:5], ['358', '502', '504', '517', '520'])

