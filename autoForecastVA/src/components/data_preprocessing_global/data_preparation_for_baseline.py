class DataPreperationForBaseline(DataDelimitingByDayAndMedicalCenter= None, DataPreparationForFeatureSelection=None, consider_selected_features=False, number_of_days_to_predict_ahead=3, find_optimal_differentiation_order=False, update_hyper_parameters_at_each_time_step=False):

    '''
    Accepts a DataDelimitingByDayAndMedicalCenter and feature selection instance, generate the rightly differenced data, the order of differentiation, and the lag order dictionary.
    day_level_medical_center_level_train_DataFrame_dictionary = {}
    day_level_medical_center_level_test_DataFrame_dictionary = {}

    day_level_medical_center_level_DataFrame_dictionary = {}

    day_level_medical_center_level_differenced_train_DataFrame_dictionary = {}
    day_level_medical_center_level_differenciation_order_dictionary = {}

    day_level_medical_center_level_lag_order_dictionary = {}

    for each day
        for each medical center
            do differentiation
            find the dataframe that contains the first day to this day (inclusive on both ends)
            set the prediction day as test dataset
            set the prediction day - number of days before as train dataset
    '''