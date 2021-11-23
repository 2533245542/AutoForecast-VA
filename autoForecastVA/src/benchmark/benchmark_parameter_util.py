import copy
class ArimaParameterUtil:
    """
    Usage: ArimaParameterUtil().total_dictionary
    """
    def __init__(self, list_of_template_dictionary=None, lazy=False):
        # inputs
        self.list_of_template_dictionary = list_of_template_dictionary
        self.lazy = lazy
        ## handle None
        if self.list_of_template_dictionary == None:
            self.list_of_template_dictionary = [{}]

        # outputs
        self.total_list_of_dictionary = None

        if not self.lazy:
            self.generate_arguments()

    def generate_arguments(self):
        self.total_list_of_dictionary = self.list_of_template_dictionary

class VarmaParameterUtil:
    """
    Usage: VarmaParameterUtil().total_dictionary
    """
    def __init__(self, list_of_template_dictionary=None, lazy=False):
        # inputs
        self.list_of_template_dictionary = list_of_template_dictionary
        self.lazy = lazy
        ## handle None
        if self.list_of_template_dictionary == None:
            self.list_of_template_dictionary = [{}]

        # outputs
        self.total_list_of_dictionary = None

        if not self.lazy:
            self.generate_arguments()

    def generate_arguments(self):
        self.total_list_of_dictionary = self.list_of_template_dictionary




class AutoVAParameterUtil:
    """
    Usage:
    AutoVAParameterUtil(list_of_template_dictionary=[{'number_of_days_to_predict_ahead': 2}], list_of_tuning_interval=[1,5,10,30], list_of_seeds=[0,1,2,3,4])
    """
    def __init__(self, list_of_template_dictionary=None, list_of_time_period_subset=None, list_of_tuning_interval=None, max_tuning_interval=9999999, generate_time_period_subset=True, generate_tuning_interval=True,
        generate_no_fine_tuning=True, generate_no_clustering=True, generate_no_clustering_and_no_feature_selection=True, list_of_seeds=None, base_name_server_port=60001, lazy=False):
        # inputs
        self.list_of_template_dictionary = list_of_template_dictionary
        self.list_of_time_period_subset = list_of_time_period_subset
        self.list_of_tuning_interval = list_of_tuning_interval
        self.max_tuning_interval = max_tuning_interval

        self.generate_time_period_subset = generate_time_period_subset
        self.generate_tuning_interval = generate_tuning_interval
        self.generate_no_fine_tuning = generate_no_fine_tuning
        self.generate_no_clustering = generate_no_clustering
        self.generate_no_clustering_and_no_feature_selection = generate_no_clustering_and_no_feature_selection

        self.list_of_seeds = list_of_seeds
        self.base_name_server_port = base_name_server_port
        self.lazy = lazy

        ## handle None mutable inputs
        if self.list_of_time_period_subset == None:
            self.list_of_time_period_subset = []  # so no a single iteration will be done and no redundant keys are added to the dictionary
        if self.list_of_template_dictionary == None:
            self.list_of_template_dictionary = [{}]
        if self.list_of_tuning_interval == None:
            self.list_of_tuning_interval = [1, 5, 10, 30]
        if self.list_of_seeds == None:
            self.list_of_seeds = [0, 1, 2, 3, 4]

        # outputs
        self.list_of_time_period_subset_dictionary = []
        self.list_of_tuning_interval_dictionary = []
        self.list_of_no_fine_tuning_dictionary = []
        self.list_of_no_clustering_dictionary = []
        self.list_of_no_clustering_and_no_feature_selection_dictionary = []
        self.total_list_of_dictionary = None
        self.seeded_total_list_of_dictionary = None
        self.port_inserted_seeded_total_list_of_dictionary = None

        if not self.lazy:
            self.generate_arguments()

    def generate_arguments(self):
        # create from templates
        if self.generate_time_period_subset:
            self.list_of_time_period_subset_dictionary = self.create_time_period_subset(list_of_template_dictionary=self.list_of_template_dictionary, list_of_time_period_subset=self.list_of_time_period_subset, max_tuning_interval=self.max_tuning_interval)
        if self.generate_tuning_interval:
            self.list_of_tuning_interval_dictionary = self.create_tuning_interval(list_of_template_dictionary=self.list_of_template_dictionary, list_of_tuning_interval=self.list_of_tuning_interval)
        if self.generate_no_fine_tuning:
            self.list_of_no_fine_tuning_dictionary = self.create_no_fine_tuning(list_of_template_dictionary=self.list_of_template_dictionary, max_tuning_interval=self.max_tuning_interval)
        if self.generate_no_clustering:
            self.list_of_no_clustering_dictionary = self.create_no_clustering(list_of_template_dictionary=self.list_of_template_dictionary, max_tuning_interval=self.max_tuning_interval)
        if self.generate_no_clustering_and_no_feature_selection:
            self.list_of_no_clustering_and_no_feature_selection_dictionary = self.create_no_clustering_and_no_feature_selection(list_of_template_dictionary=self.list_of_template_dictionary, max_tuning_interval=self.max_tuning_interval)

        # combine to one list
        self.total_list_of_dictionary = self.list_of_time_period_subset_dictionary + self.list_of_tuning_interval_dictionary + self.list_of_no_fine_tuning_dictionary + self.list_of_no_clustering_dictionary + self.list_of_no_clustering_and_no_feature_selection_dictionary

        # replicate
        self.seeded_total_list_of_dictionary = self.replicate_by_seeds(list_of_parameter_to_modified_value_dictionary=self.total_list_of_dictionary, list_of_seeds=self.list_of_seeds)

        # insert
        self.port_inserted_seeded_total_list_of_dictionary = self.insert_unique_name_server_port(list_of_parameter_to_modified_value_dictionary=self.seeded_total_list_of_dictionary, base_nameserver_port=self.base_name_server_port)

    @staticmethod
    def create_time_period_subset(list_of_template_dictionary, list_of_time_period_subset=None, max_tuning_interval=9999999):
        '''Add one time period to each template dictionary'''
        list_of_created_dictionary = []
        for template_dictionary in list_of_template_dictionary:
            for time_period_interval in list_of_time_period_subset:
                created_dictionary = copy.deepcopy(template_dictionary)
                created_dictionary.update({'time_period_subset': time_period_interval, 'tuning_frequency': max_tuning_interval})
                list_of_created_dictionary.append(created_dictionary)
        return list_of_created_dictionary

    @staticmethod
    def create_tuning_interval(list_of_template_dictionary, list_of_tuning_interval=None):
        """Create from template and generate tuning intervals."""
        list_of_created_dictionary = []
        for template_dictionary in list_of_template_dictionary:
            for interval in list_of_tuning_interval:
                created_dictionary = copy.deepcopy(template_dictionary)
                created_dictionary.update({'tuning_frequency': interval})
                list_of_created_dictionary.append(created_dictionary)
        return list_of_created_dictionary

    @staticmethod
    def create_no_fine_tuning(list_of_template_dictionary, max_tuning_interval=9999999):
        """Create from template and generate no fine tuning. The general model is directly used by each clinic."""
        list_of_created_dictionary = []
        for template_dictionary in list_of_template_dictionary:
            created_dictionary = copy.deepcopy(template_dictionary)
            created_dictionary.update({'do_fine_tuning': False, 'tuning_frequency': max_tuning_interval})
            list_of_created_dictionary.append(created_dictionary)
        return list_of_created_dictionary

    @staticmethod
    def create_no_clustering(list_of_template_dictionary, max_tuning_interval=9999999):
        """Create from template and generate no clustering. Fine-tune one model for each clinic per day."""
        list_of_created_dictionary = []
        for template_dictionary in list_of_template_dictionary:
            created_dictionary = copy.deepcopy(template_dictionary)
            created_dictionary.update({'do_clustering': False, 'do_general_model_training': False, 'tuning_frequency': max_tuning_interval})
            list_of_created_dictionary.append(created_dictionary)
        return list_of_created_dictionary

    @staticmethod
    def create_no_clustering_and_no_feature_selection(list_of_template_dictionary, max_tuning_interval=9999999):
        """Create from template and generate no clustering and no feature selection."""
        list_of_created_dictionary = []
        for template_dictionary in list_of_template_dictionary:
            created_dictionary = copy.deepcopy(template_dictionary)
            created_dictionary.update({'do_clustering': False, 'do_general_model_training': False, 'do_feature_selection': False, 'tuning_frequency': max_tuning_interval})
            list_of_created_dictionary.append(created_dictionary)
        return list_of_created_dictionary

    @staticmethod
    def replicate_by_seeds(list_of_parameter_to_modified_value_dictionary, list_of_seeds=None):
        """Replicate each input dictionary by the number of seeds."""
        list_of_replicated_dictionary = []
        for original_dictionary in list_of_parameter_to_modified_value_dictionary:
            for seed in list_of_seeds:
                replicated_dictionary = copy.deepcopy(original_dictionary)
                replicated_dictionary['hyper_parameter_space_seed'] = seed
                replicated_dictionary['numpy_seed'] = seed
                replicated_dictionary['neural_network_training_seed'] = seed
                list_of_replicated_dictionary.append(replicated_dictionary)

        return list_of_replicated_dictionary

    @staticmethod
    def insert_unique_name_server_port(list_of_parameter_to_modified_value_dictionary, base_nameserver_port=60001):
        """Add a unique name server port for each input dictionary"""
        for original_dictionary in list_of_parameter_to_modified_value_dictionary:
            original_dictionary['nameserver_port'] = base_nameserver_port
            original_dictionary['nameserver_run_id'] = 'run_id' + str(base_nameserver_port)
            base_nameserver_port += 1
        return list_of_parameter_to_modified_value_dictionary





