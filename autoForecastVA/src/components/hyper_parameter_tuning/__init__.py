'''
Given a medical_center_level_input_output_DataFrame_list_dictionary (generated by day_level_cluster_level_medical_center_level_input_output_DataFrame_list_dictionary[day][cluster]) and the number of test day in a day_level_DataFrame, extract X where the first dimension is the number of input_output_DataFrame, the second dimension is the number of time steps and the third dimension is the number of features; we also extract Y where the first dimension is the number of input_output_DataFrame, the second dimension is the number of time steps (one in this day, the prediction day) and the third dimension is the number of features (1 in this case, the number of COVID-19 related calls). The generated X and Y will be shuffled while keeping the mappings between X and Y unchanged.

For each automated machine learning iteration, a hyper-parameter value combination will be generated, and along with the X_train, Y_train, X_test and Y_test, it will be fed into a LSTM network building component, the LSTM network returns its loss on Y_test.

This file eventually outputs an optimal hyper-parameter value combination.
'''