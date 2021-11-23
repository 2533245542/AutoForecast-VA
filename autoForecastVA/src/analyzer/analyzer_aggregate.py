from autoForecastVA.src.data_caching.variable_load_and_save import VariableSaverAndLoader
import numpy as np
import pandas as pd


class AnalyzerAggregate():
    """Merge multiple analyzer results into one."""
    def __init__(self, list_of_result_files=None, list_of_aggregatable_variable_names=None, aggregatable_variable_name_to_aggregate_way_mapping=None, lazy=False):
        # inputs
        self.list_of_result_files = list_of_result_files
        self.list_of_aggregatable_variable_names = list_of_aggregatable_variable_names
        self.aggregatable_variable_name_to_aggregate_way_mapping = aggregatable_variable_name_to_aggregate_way_mapping

        # outputs
        self.result_file_to_result_mapping = None
        self.variable_name_to_aggregated_variable_mapping = None

        if not lazy:
            if self.aggregatable_variable_name_to_aggregate_way_mapping == None:  # default to use mean aggregation
                self.aggregatable_variable_name_to_aggregate_way_mapping = {}
                for aggregatable_variable_name in list_of_aggregatable_variable_names:
                    self.aggregatable_variable_name_to_aggregate_way_mapping[aggregatable_variable_name] = 'mean'


            self.result_file_to_result_mapping = self.read_result_files(list_of_result_files=self.list_of_result_files)
            self.variable_name_to_aggregated_variable_mapping = self.do_aggregation(
                list_of_results=list(self.result_file_to_result_mapping.values()), list_of_aggregatable_variable_names=self.list_of_aggregatable_variable_names, aggregatable_variable_name_to_aggregate_way_mapping=self.aggregatable_variable_name_to_aggregate_way_mapping)

    @staticmethod
    def read_result_files(list_of_result_files):
        """Read results from all result files."""
        result_file_to_result_mapping = {}
        for result_file_name in list_of_result_files:
            result = VariableSaverAndLoader(load=True, file_name=result_file_name).list_of_loaded_variables[0]
            result_file_to_result_mapping[result_file_name] = result
        return result_file_to_result_mapping

    @staticmethod
    def do_aggregation(list_of_results, list_of_aggregatable_variable_names, aggregatable_variable_name_to_aggregate_way_mapping):
        """Do aggregation for ach aggregatable variable over all results."""
        variable_name_to_aggregated_variable_mapping = {}
        for aggregatable_variable_name in list_of_aggregatable_variable_names:
            # get the seeded clones of the variable
            list_of_seeded_clones = []
            for result in list_of_results:
                list_of_seeded_clones.append(getattr(result, aggregatable_variable_name))

            aggregation_way = aggregatable_variable_name_to_aggregate_way_mapping[aggregatable_variable_name]

            variable_to_aggregate_is_dataframe = all([isinstance(clone, pd.DataFrame) for clone in list_of_seeded_clones])
            variable_to_aggregate_is_float = all([isinstance(clone, float) for clone in list_of_seeded_clones])

            if variable_to_aggregate_is_dataframe:  # use aggregate over a variable dataframe
                variable_name_to_aggregated_variable_mapping[aggregatable_variable_name] = AnalyzerAggregate.aggregate_over_a_variable_dataframe( list_of_aggregatable_variable_clones=list_of_seeded_clones, aggregation_way=aggregation_way)

            if variable_to_aggregate_is_float:
                variable_name_to_aggregated_variable_mapping[aggregatable_variable_name+'_mean'], variable_name_to_aggregated_variable_mapping[aggregatable_variable_name+'_std'] = AnalyzerAggregate.aggregate_over_a_variable_float(list_of_aggregatable_variable_clones=list_of_seeded_clones)

        return variable_name_to_aggregated_variable_mapping

    @staticmethod
    def aggregate_over_a_variable_dataframe(list_of_aggregatable_variable_clones, group_keys=None, aggregation_way='mean'):
        """Aggregate over a variable from a list of seeded clones."""
        concated_dataframe = pd.concat(list_of_aggregatable_variable_clones, axis=0)
        aggregated_dataframe = concated_dataframe.groupby(by=['day', 'clinic']).mean()
        aggregated_dataframe_index_reset = aggregated_dataframe.reset_index()
        completed_dataframe = aggregated_dataframe_index_reset.set_index('day')

        return completed_dataframe

    @staticmethod
    def aggregate_over_a_variable_float(list_of_aggregatable_variable_clones, group_keys=None, aggregation_way='mean'):
        """Aggregate over a variable from a list of floats."""
        return np.mean(list_of_aggregatable_variable_clones), np.std(list_of_aggregatable_variable_clones)