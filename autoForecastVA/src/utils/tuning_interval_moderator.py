import numpy as np
class OperationIntervalModerator():
    def __init__(self, days, operation_interval=1, lazy=False):
        # inputs
        self.days = days
        self.operation_interval = operation_interval  # do tuning for every day when tuning_interval=1

        # outputs
        self.sorted_days = None
        self.total_number_of_days = None
        self.list_of_operation_idle_day_index_list = None
        self.operation_day_to_idle_day_list_dictionary = {}   # an operation day has a list of (followed) idles days
        self.day_to_operation_day_dictionary = {}   # each day has an operation day

        if not lazy:
            self.calculate_operation_and_idle_days()

    def calculate_operation_and_idle_days(self):
        '''
        Operation day: the day that we need to do operation (clustering, feature selection, tuning)
        Idle days: days between two operation days; do not need operation
        One operation interval: one operation day + a list of idle days
        '''
        # days are splited to lists of opeation interval days; tuning day is the first day of a list of operation interval day
        self.sorted_days = sorted(self.days)
        self.total_number_of_days = len(self.sorted_days)

        operation_day_index_list = list(range(0, self.total_number_of_days, self.operation_interval))
        self.list_of_operation_idle_day_index_list = np.split(ary=np.arange(self.total_number_of_days), indices_or_sections=operation_day_index_list)[1:]

        for operation_idle_day_index_list in self.list_of_operation_idle_day_index_list:
            operation_idle_day_list = [self.sorted_days[day_index] for day_index in operation_idle_day_index_list]

            operation_day = operation_idle_day_list[0]

            self.operation_day_to_idle_day_list_dictionary[operation_day] = operation_idle_day_list[1:]

            for day in operation_idle_day_list:
                self.day_to_operation_day_dictionary[day] = operation_day  # each day will have a tuning day

    def day_is_operation_day(self, day):
        return day in self.operation_day_to_idle_day_list_dictionary.keys()

    def get_operation_day_of_a_day(self, day):
        return self.day_to_operation_day_dictionary[day]
