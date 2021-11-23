from autoForecastVA.src.analyzer.analyzer_pipeline_autoForecastVA import AnalyzerPipeLineAutoForecastVA
from autoForecastVA.src.data_caching.variable_load_and_save import VariableSaverAndLoader
from autoForecastVA.src.utils.dictionary_wrangling import DictionaryWrangling
from sklearn import metrics
import pandas as pd
import seaborn as sns




class AnalyzerPipeLineArima:
    def __init__(self, pipelineArima, lazy=False):
        """Inspect a pipelineArima and produce convinient data structures for data science analysis."""
        # inputs
        self.pipelineArima = pipelineArima
        self.lazy = lazy

        # outputs
        self.dataImputationAndAveraging = pipelineArima.dataImputationAndAveraging
        self.dataDelimitingByDayAndMedicalCenter = pipelineArima.dataDelimitingByDayAndMedicalCenter
        self.dataCluseteringAndNormalization = pipelineArima.dataCluseteringAndNormalization
        self.dataPreparationForFeatureSelection = pipelineArima.dataPreparationForFeatureSelection
        self.featureSelectionGlobal = pipelineArima.featureSelectionGlobal

        if not self.lazy:
            self.day_level_cluster_level_medical_center_list_dictionary = self.get_day_level_cluster_level_medical_center_list_dictionary()
            self.day_level_medical_center_to_cluster_dictionary = self.get_day_level_medical_center_to_cluster_dictionary()

            self.day_level_agency_level_feature_index_list_dictionary = self.featureSelectionGlobal.day_level_agency_level_feature_index_list_dictionary
            self.day_level_agency_level_total_p_value_list_dictionary = self.featureSelectionGlobal.day_level_agency_level_total_p_value_list_dictionary

            self.day_to_clinic_to_index_to_train_prediction_target_dataframe = self.get_day_to_clinic_to_index_to_train_prediction_target_dataframe(self.pipelineArima)

            self.day_to_clinic_to_test_prediction_target_dataframe = self.get_day_to_clinic_to_test_prediction_target_dataframe(self.pipelineArima)

            self.day_to_clinic_to_index_to_train_prediction_dataframe = self.get_day_to_clinic_to_index_to_train_prediction_dataframe(self.pipelineArima)

            self.day_to_clinic_to_test_prediction_dataframe = self.get_day_to_clinic_to_test_prediction_dataframe(self.pipelineArima)

            self.day_to_clinic_to_test_prediction_loss_dataframe = self.get_day_to_clinic_to_test_prediction_loss_dataframe(day_to_clinic_to_test_prediction_target_dataframe=self.day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe=self.day_to_clinic_to_test_prediction_dataframe)
            self.clinic_to_test_prediction_loss_dataframe = self.get_clinic_to_test_prediction_loss_dataframe(day_to_clinic_to_test_prediction_target_dataframe=self.day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe=self.day_to_clinic_to_test_prediction_dataframe)
            self.test_prediction_loss = self.get_test_prediction_loss(day_to_clinic_to_test_prediction_target_dataframe=self.day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe=self.day_to_clinic_to_test_prediction_dataframe)

    def get_day_level_cluster_level_medical_center_list_dictionary(self):
        return self.dataCluseteringAndNormalization.day_level_cluster_level_medical_center_list_dictionary

    def get_day_level_medical_center_to_cluster_dictionary(self):
        return self.dataCluseteringAndNormalization.day_level_medical_center_to_cluster_dictionary

    @staticmethod
    def get_day_to_clinic_to_index_to_train_prediction_target_dataframe(pipelineArima):
        day_to_clinic_to_list_of_train_prediction_target_dictionary = AnalyzerPipeLineAutoForecastVA.transform_value_of_two_level_dictionary(first_level_second_level_item_dictionary=pipelineArima.day_level_clinic_level_arimaModel_dictionary, value_transform_function=lambda arimaModel: arimaModel.train_dataframe.iloc[:, 0].tolist())
        dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_key_then_list_dictionary_to_others(first_level_second_level_list_dictionary=day_to_clinic_to_list_of_train_prediction_target_dictionary)
        dataframe = AnalyzerPipeLineAutoForecastVA.reformat_dataframe(input_dataframe=dataframe, new_column_names=['day', 'clinic', 'index', 'train_prediction_target'], name_of_index_column='day')
        return dataframe

    @staticmethod
    def get_day_to_clinic_to_test_prediction_target_dataframe(pipelineArima):
        dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_dictionary_to_others(first_level_second_level_item_dictionary=pipelineArima.day_level_clinic_level_arimaModel_dictionary, item_converter=lambda arimaModel: arimaModel.test_dataframe.iloc[0, 0])
        dataframe = AnalyzerPipeLineAutoForecastVA.reformat_dataframe(input_dataframe=dataframe, new_column_names=['day', 'clinic', 'test_prediction_target'],  name_of_index_column='day')
        return dataframe

    @staticmethod
    def get_day_to_clinic_to_index_to_train_prediction_dataframe(pipelineArima):
        day_to_clinic_to_list_of_train_prediction_dictionary = AnalyzerPipeLineAutoForecastVA.transform_value_of_two_level_dictionary(first_level_second_level_item_dictionary=pipelineArima.day_level_clinic_level_arimaModel_dictionary, value_transform_function=lambda arimaModel: arimaModel.train_predictions.iloc[:, 0].tolist())

        dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_key_then_list_dictionary_to_others(first_level_second_level_list_dictionary=day_to_clinic_to_list_of_train_prediction_dictionary)
        dataframe = AnalyzerPipeLineAutoForecastVA.reformat_dataframe(input_dataframe=dataframe, new_column_names=['day', 'clinic', 'index', 'train_prediction'], name_of_index_column='day')
        return dataframe

    @staticmethod
    def get_day_level_medical_center_level_fine_tuned_model_train_last_epoch_loss_dictionary(day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary):
        day_level_clinic_level_train_loss_dictionary = DictionaryWrangling.create_empty_nested_dictionary(
            keys=day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary.keys())

        for day in sorted(day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary.keys()):
            for clinic in day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary[day].keys():
                day_level_clinic_level_train_loss_dictionary[day][clinic] = day_level_medical_center_level_fine_tuned_model_train_epoch_loss_list_dictionary[day][clinic][-1]
        return day_level_clinic_level_train_loss_dictionary

    @staticmethod
    def get_day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe(day_level_medical_center_level_fine_tuned_model_train_last_epoch_loss_dictionary):
        day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_dictionary_to_others(day_level_medical_center_level_fine_tuned_model_train_last_epoch_loss_dictionary)
        day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe.columns = ['day', 'clinic', 'loss']
        day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe = day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe.set_index('day')
        return day_to_clinic_to_fine_tuned_model_last_epoch_loss_dataframe

    @staticmethod
    def get_day_to_clinic_to_test_prediction_dataframe(pipelineArima):
        dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_dictionary_to_others(first_level_second_level_item_dictionary=pipelineArima.day_level_clinic_level_arimaModel_dictionary, item_converter=lambda arimaModel: arimaModel.test_predictions.iloc[0, 0])
        dataframe = AnalyzerPipeLineAutoForecastVA.reformat_dataframe(input_dataframe=dataframe, new_column_names=['day', 'clinic', 'test_prediction'], name_of_index_column='day')
        return dataframe

    @staticmethod
    def get_day_to_clinic_to_test_prediction_loss_dataframe(day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe, loss_function=metrics.mean_squared_error):
        day_to_clinic_to_test_prediction_loss_dataframe = pd.merge(left=day_to_clinic_to_test_prediction_target_dataframe, right=day_to_clinic_to_test_prediction_dataframe, on=['day', 'clinic'], validate='one_to_one')

        list_of_daily_clinic_loss = []
        for i in range(day_to_clinic_to_test_prediction_loss_dataframe.shape[0]):
            list_of_daily_clinic_loss.append(loss_function( day_to_clinic_to_test_prediction_loss_dataframe.iloc[[i],].test_prediction_target.tolist(), day_to_clinic_to_test_prediction_loss_dataframe.iloc[[i],].test_prediction.tolist()))

        day_to_clinic_to_test_prediction_loss_dataframe['loss'] = list_of_daily_clinic_loss
        return day_to_clinic_to_test_prediction_loss_dataframe

    @staticmethod
    def get_clinic_to_test_prediction_loss_dataframe(day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe, loss_function=metrics.mean_squared_error):
        '''usage
        clinic_to_test_prediction_loss_dataframe = AnalyzerPipeLineAutoForecastVA.get_clinic_to_test_prediction_loss_dataframe( day_to_clinic_to_test_prediction_target_dataframe=day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe=day_to_clinic_to_test_prediction_dataframe, loss_function=metrics.mean_squared_error)
        '''
        day_to_clinic_to_test_prediction_loss_dataframe = pd.merge( left=day_to_clinic_to_test_prediction_target_dataframe, right=day_to_clinic_to_test_prediction_dataframe, on=['day', 'clinic'], validate='one_to_one')

        list_of_clinic = []
        list_of_loss = []
        for clinic in day_to_clinic_to_test_prediction_loss_dataframe.clinic.unique():
            day_to_test_prediction_loss_dataframe = day_to_clinic_to_test_prediction_loss_dataframe[ day_to_clinic_to_test_prediction_loss_dataframe.clinic == clinic]
            single_clinic_test_prediction_loss = loss_function( day_to_test_prediction_loss_dataframe.test_prediction_target.tolist(), day_to_test_prediction_loss_dataframe.test_prediction.tolist())
            list_of_clinic.append(clinic)
            list_of_loss.append(single_clinic_test_prediction_loss)

        clinic_to_test_prediction_loss_dataframe = pd.DataFrame({'clinic': list_of_clinic, 'loss': list_of_loss})
        return clinic_to_test_prediction_loss_dataframe

    @staticmethod
    def get_test_prediction_loss(day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe, loss_function=metrics.mean_squared_error):
        '''usage
        test_prediction_loss = AnalyzerPipeLineAutoForecastVA.get_test_prediction_loss( day_to_clinic_to_test_prediction_target_dataframe=day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe=day_to_clinic_to_test_prediction_dataframe)
        '''

        day_to_clinic_to_test_prediction_loss_dataframe = pd.merge(left=day_to_clinic_to_test_prediction_target_dataframe, right=day_to_clinic_to_test_prediction_dataframe, on=['day', 'clinic'], validate='one_to_one')
        test_prediction_loss = loss_function(day_to_clinic_to_test_prediction_loss_dataframe.test_prediction_target, day_to_clinic_to_test_prediction_loss_dataframe.test_prediction)
        return test_prediction_loss


    # helper functions
    '''loopers'''

    '''make figures'''
    @staticmethod
    def make_figure_for_clusterization_trend(day_level_medical_center_to_cluster_dictionary, rotation_degree=30, height=20.27, aspect=11.7 / 20.27):
        '''
        def make_figure_for_clusterization_trend
        input:
        day_level_medical_center_to_cluster_dictionary
        rotation_degree

        output:
        a figure
        steps:

        usage:
        g = make_figure_for_clusterization_trend(day_level_medical_center_to_cluster_dictionary)

        '''

        day_to_clinic_to_cluster_dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_dictionary_to_others( first_level_second_level_item_dictionary=day_level_medical_center_to_cluster_dictionary)
        day_to_clinic_to_cluster_dataframe.columns = ['day', 'clinic', 'cluster']
        day_to_clinic_to_cluster_dataframe = day_to_clinic_to_cluster_dataframe.set_index('day')

        figure_for_clusterzation_trend = sns.relplot(data=day_to_clinic_to_cluster_dataframe, x='day', y='clinic', hue='cluster', height=height, aspect=aspect)
        figure_for_clusterzation_trend.set_xticklabels(rotation=rotation_degree)
        return figure_for_clusterzation_trend

    @staticmethod
    def make_figure_for_cluster_size_trend(day_level_cluster_level_medical_center_list_dictionary):
        '''
        def make_figure_for_cluster_size_trend
        inputs:
        day_level_cluster_level_medical_center_list_dictionary
        outputs:
        a figure
        steps:
        usage:
        figure_for_cluster_size_trend = make_figure_for_cluster_size_trend(day_level_cluster_level_medical_center_list_dictionary=day_level_cluster_level_medical_center_list_dictionary)
        figure_for_cluster_size_trend.savefig('figure_for_cluster_size_trend.png')
        '''

        day_to_cluster_to_size_dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_dictionary_to_others(first_level_second_level_item_dictionary=day_level_cluster_level_medical_center_list_dictionary, item_converter=lambda x: len(x), return_type='dataframe')
        day_to_cluster_to_size_dataframe.columns = ['day', 'cluster', 'size']
        day_to_cluster_to_size_dataframe = day_to_cluster_to_size_dataframe.set_index('day')
        plot_df = day_to_cluster_to_size_dataframe.pivot_table(index='day', columns='cluster', values='size').fillna(0)
        plot_df.index = plot_df.index.strftime('%Y-%m-%d')
        ax = plot_df.plot(kind='bar', stacked=True)
        return ax.figure


    @staticmethod
    def make_figure_for_selected_features_trend(day_level_agency_level_feature_index_list_dictionary,
                                                rotation_degree=30):
        day_to_cluster_to_index_of_feature_index_to_feature_index_dataframe = AnalyzerPipeLineAutoForecastVA.convert_nested_key_then_list_dictionary_to_others( day_level_agency_level_feature_index_list_dictionary)
        day_to_cluster_to_index_of_feature_index_to_feature_index_dataframe.columns = ['day', 'cluster', 'index_of_feature_index', 'feature_index']
        day_to_cluster_to_feature_index_dataframe = day_to_cluster_to_index_of_feature_index_to_feature_index_dataframe[ ['day', 'cluster', 'feature_index']]
        day_to_cluster_to_feature_index_dataframe = day_to_cluster_to_feature_index_dataframe.set_index('day')

        figure_for_selected_features_trends = sns.relplot(data=day_to_cluster_to_feature_index_dataframe, x='day', y='feature_index', row='cluster')
        figure_for_selected_features_trends.set_xticklabels(rotation=rotation_degree)
        return figure_for_selected_features_trends

    @staticmethod
    def make_figure_for_number_of_selected_features_trend(day_level_agency_level_feature_index_list_dictionary,
                                                          rotation_degree=30):
        day_to_cluster_to_number_of_selected_features = AnalyzerPipeLineAutoForecastVA.convert_nested_dictionary_to_others( day_level_agency_level_feature_index_list_dictionary, item_converter=lambda x: len(x))
        day_to_cluster_to_number_of_selected_features.columns = ['day', 'cluster', 'number_of_selected_features']
        day_to_cluster_to_feature_index_dataframe = day_to_cluster_to_number_of_selected_features.set_index('day')

        figure_for_number_of_selected_features_trend = sns.relplot(data=day_to_cluster_to_feature_index_dataframe, x='day', y='number_of_selected_features', row='cluster')
        figure_for_number_of_selected_features_trend.set_xticklabels(rotation=rotation_degree)
        return figure_for_number_of_selected_features_trend

    @staticmethod
    def make_figure_for_hyper_parameter_value_combination_trend(day_to_cluster_to_hyper_parameter_to_value_combination_dataframe, rotation_degree=30, kind='line'):
        ''' kind = 'scatter' or 'line' '''
        day_to_cluster_to_hyper_parameter_to_value_combination_dataframe = day_to_cluster_to_hyper_parameter_to_value_combination_dataframe.reset_index()
        figure_for_hyper_parameter_value_combination_trend = sns.relplot( data=day_to_cluster_to_hyper_parameter_to_value_combination_dataframe, x='day', y='value', kind=kind, row='hyper_parameter', facet_kws={'sharey': False, 'sharex': True})
        figure_for_hyper_parameter_value_combination_trend.set_xticklabels(rotation=rotation_degree)
        return figure_for_hyper_parameter_value_combination_trend

    @staticmethod
    def make_figure_for_all_clinic_test_target_and_prediction(day_to_clinic_to_test_prediction_dataframe, day_to_clinic_to_test_prediction_target_dataframe, rotation_degree=30):
        '''
        inputs:
        day_to_clinic_to_test_prediction_dataframe
        day_to_clinic_to_test_prediction_target_dataframe

        outputs:
        a figure

        steps:
        for both dataframe,
        add a column,
        reset index
        rename the column names to  ['day', 'clinic', 'value', 'target_or_prediction']
        concate them to create plot_df

        g = sns.FacetGrid(plot_df, col='clinic', hue='target_or_prediction')
        g.map(plt.scatter, "day", "value")
        g.add_legend()
        plt.show()

        usage:
        plot_predictions_all_clinic(result_dictionary['day_to_clinic_to_test_prediction_target_dataframe'], result_dictionary['day_to_clinic_to_test_prediction_dataframe'])
        '''

        day_to_clinic_to_test_prediction_target_dataframe = day_to_clinic_to_test_prediction_target_dataframe.reset_index()  # HACK must do reset index first to create a copy of the dataframe
        day_to_clinic_to_test_prediction_target_dataframe['target_or_prediction'] = 'target'
        day_to_clinic_to_test_prediction_target_dataframe.columns = ['day', 'clinic', 'value', 'target_or_prediction']

        day_to_clinic_to_test_prediction_dataframe = day_to_clinic_to_test_prediction_dataframe.reset_index()  # HACK
        day_to_clinic_to_test_prediction_dataframe['target_or_prediction'] = 'prediction'
        day_to_clinic_to_test_prediction_dataframe.columns = ['day', 'clinic', 'value', 'target_or_prediction']

        plot_df = pd.concat([day_to_clinic_to_test_prediction_target_dataframe, day_to_clinic_to_test_prediction_dataframe], axis=0)

        g = sns.relplot(data=plot_df, x='day', y='value', hue='target_or_prediction', col='clinic', col_wrap=10, kind='line')
        g.set_xticklabels(rotation=rotation_degree)

        return g

    @staticmethod
    def make_figure_for_test_prediction_loss(day_to_clinic_to_test_prediction_loss_dataframe, col_wrap=10, rotation_degree=30):
        '''
        usage:
        figure_for_test_prediction_loss = AnalyzerPipeLineAutoForecastVA.make_figure_for_test_prediction_loss(day_to_clinic_to_test_prediction_loss_dataframe=day_to_clinic_to_test_prediction_loss_dataframe, col_wrap=10, rotation_degree=30)
        figure_for_test_prediction_loss.savefig('figure_for_test_prediction_loss.png')
        '''
        plot_dataframe = day_to_clinic_to_test_prediction_loss_dataframe.melt(id_vars='clinic', value_vars=['test_prediction_target', 'test_prediction', 'loss'], ignore_index=False)
        figure_for_test_prediction_loss = sns.relplot(data=plot_dataframe, x='day', y='value', hue='variable', col='clinic', col_wrap=col_wrap, kind='line')
        figure_for_test_prediction_loss.set_xticklabels(rotation=rotation_degree)
        return figure_for_test_prediction_loss

    @staticmethod
    def save_analyzer_result_in_a_memory_efficient_way(pipelineArima, file_name='data/memory_efficient_analyzer.dat'):
        """Remove the data heavy part of a pipeline, and store the rest."""
        analyzerPipeLineArima = AnalyzerPipeLineArima(pipelineArima)
        analyzerPipeLineArima.pipelineArima = None
        analyzerPipeLineArima.dataImputationAndAveraging = None
        analyzerPipeLineArima.dataDelimitingByDayAndMedicalCenter = None
        analyzerPipeLineArima.dataCluseteringAndNormalization = None
        analyzerPipeLineArima.dataPreparationForFeatureSelection = None
        analyzerPipeLineArima.featureSelectionGlobal = None

        VariableSaverAndLoader(save=True, list_of_variables_to_save=[analyzerPipeLineArima], file_name=file_name)
