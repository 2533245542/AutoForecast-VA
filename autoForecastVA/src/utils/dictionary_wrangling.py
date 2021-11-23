import pandas as pd
class DictionaryWrangling():
    @staticmethod
    def create_empty_nested_dictionary(keys=None):
        empty_nested_dictionary = {}
        for key in keys:
            empty_nested_dictionary[key] = {}
        return empty_nested_dictionary

    @staticmethod
    def looper_double(first_level_second_level_item_dictionary, value_getter_function):
        first_level_second_level_value_dictionary = DictionaryWrangling.create_empty_nested_dictionary(first_level_second_level_item_dictionary.keys())
        for first_level_key in first_level_second_level_item_dictionary.keys():
            for second_level_key in first_level_second_level_item_dictionary[first_level_key].keys():
                item = first_level_second_level_item_dictionary[first_level_key][second_level_key]
                value = value_getter_function(item)
                first_level_second_level_value_dictionary[first_level_key][second_level_key] = value

        return first_level_second_level_value_dictionary

    '''convert dictionary to dataframe'''
    @staticmethod
    def convert_nested_dictionary_to_others(first_level_second_level_item_dictionary, first_level_key_converter=lambda x: x, second_level_key_converter=lambda x: x, item_converter=lambda x: x, return_type='dataframe'):
        '''
        return_type: 'dictionary', 'dataframe'
        '''
        list_of_first_level_key = []
        list_of_second_level_key = []
        list_of_items = []
        for first_level_key in first_level_second_level_item_dictionary.keys():
            for second_level_key in first_level_second_level_item_dictionary[first_level_key].keys():
                list_of_first_level_key.append(first_level_key_converter(first_level_key))
                list_of_second_level_key.append(second_level_key_converter(second_level_key))
                list_of_items.append(item_converter(first_level_second_level_item_dictionary[first_level_key][second_level_key]))
        result_dictionary = {'first_level_key': list_of_first_level_key, 'second_level_key': list_of_second_level_key, 'item': list_of_items}
        if return_type == 'dictionary':
            return result_dictionary
        elif return_type == 'dataframe':
            return pd.DataFrame(result_dictionary)
        else:
            raise NotImplementedError


    @staticmethod
    def convert_nested_key_then_list_dictionary_to_others(first_level_second_level_list_dictionary, first_level_key_converter=lambda x: x, second_level_key_converter=lambda x: x, sub_item_converter=lambda x: x, return_type='dataframe'):   # for list-like item
        list_of_first_level_key = []
        list_of_second_level_key = []
        list_of_index = []
        list_of_sub_items = []  # a sub_item is like an elment of a list
        for first_level_key in first_level_second_level_list_dictionary.keys():
            for second_level_key in first_level_second_level_list_dictionary[first_level_key].keys():
                for index, sub_item in enumerate(first_level_second_level_list_dictionary[first_level_key][second_level_key]):
                    list_of_first_level_key.append(first_level_key_converter(first_level_key))
                    list_of_second_level_key.append(second_level_key_converter(second_level_key))
                    list_of_index.append(index)
                    list_of_sub_items.append(sub_item_converter(sub_item))
        result_dictionary = {'first_level_key': list_of_first_level_key, 'second_level_key': list_of_second_level_key, 'index': list_of_index, 'sub_item': list_of_sub_items}
        if return_type == 'dictionary':
            return result_dictionary
        elif return_type == 'dataframe':
            return pd.DataFrame(result_dictionary)
        else:
            raise NotImplementedError

    @staticmethod
    def convert_triple_nested_dictionary_to_others(first_level_second_level_third_level_item_dictionary, first_level_key_converter=lambda x: x, second_level_key_converter=lambda x: x, third_level_key_converter=lambda x: x, item_converter=lambda x: x, return_type='dataframe'):   # for hyperpaemter value combination
        list_of_first_level_key = []
        list_of_second_level_key = []
        list_of_third_level_key = []
        list_of_items = []
        for first_level_key in first_level_second_level_third_level_item_dictionary.keys():
            for second_level_key in first_level_second_level_third_level_item_dictionary[first_level_key].keys():
                for third_level_key in first_level_second_level_third_level_item_dictionary[first_level_key][second_level_key].keys():
                    list_of_first_level_key.append(first_level_key_converter(first_level_key))
                    list_of_second_level_key.append(second_level_key_converter(second_level_key))
                    list_of_third_level_key.append(third_level_key_converter(third_level_key))
                    list_of_items.append(item_converter(first_level_second_level_third_level_item_dictionary[first_level_key][second_level_key][third_level_key]))
        result_dictionary = {'first_level_key': list_of_first_level_key, 'second_level_key': list_of_second_level_key, 'third_level_key': list_of_third_level_key, 'item': list_of_items}
        if return_type == 'dictionary':
            return result_dictionary
        elif return_type == 'dataframe':
            return pd.DataFrame(result_dictionary)
        else:
            raise NotImplementedError

    @staticmethod
    def convert_list_of_one_level_dictionary_to_dataframe(list_of_input_dictionary, list_of_column_name, name_of_identifier_column='identifier'):
        '''Make each dictionary a column in the dataframe. All dictionaries should have the same keys.'''
        # get the identifier column
        identifier_column = list(list_of_input_dictionary[0].keys())
        list_of_dictionary_columns = []

        # get the value columns
        for input_dictionary in list_of_input_dictionary:
            list_of_dictionary_columns.append(list(input_dictionary.values()))

        # creating a dataframe from the identifier and value columns
        template_for_creating_dataframe = {}
        template_for_creating_dataframe[name_of_identifier_column] = identifier_column
        for column_name, column in zip(list_of_column_name, list_of_dictionary_columns):
            template_for_creating_dataframe[column_name] = column

        converted_dataframe = pd.DataFrame(template_for_creating_dataframe)
        return converted_dataframe