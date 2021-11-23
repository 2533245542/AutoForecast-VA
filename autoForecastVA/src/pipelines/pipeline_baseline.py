class baseline_VAR():
    '''
    day_level_medical_center_level_model_dictionary = {}
    day_level_medical_center_level_test_value_dictionary = {}
    day_level_medical_center_level_predicted_value_dictionary = {}
    day_level_medical_center_level_MAE_dictionary = {}
    medical_center_level_MAE_dictionary = {}
    for each day
        medical_center_level_model_dictionary = {}
        for medical_center
            difference the train dataset to stationary and find the order of the VAR model
            build VAR model from the train dataset
            medical_center_level_model_dictionary[medical_center] = model
        day_level_medical_center_level_model_dictionary[day] = medical_center_level_model_dictionary

    for each day
        medical_center_level_test_value_dictionary = {}
        medical_center_level_predicted_value_dictionary = {}
        day_level_medical_center_level_MAE_dictionary = {}
        for medical_center
            build VAR model from the train dataset
            medical_center_level_model_dictionary[medical_center] = model

        day_level_medical_center_level_model_dictionary[day] = medical_center_level_model_dictionary

    for each medical_center
        MAE_list = []
        for each day
            MAE_list.append(day_level_medical_center_level_MAE_dictionary[day][medical_center])
        medical_center_level_MAE_dictionary[medical_center] = sum(MAE_list)/len(MAE_list)
    '''

