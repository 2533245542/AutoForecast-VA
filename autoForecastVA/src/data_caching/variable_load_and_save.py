import pickle
from pathlib import Path
import time
from os import path

class VariableSaverAndLoader():
    def __init__(self, list_of_variables_to_save=None, save=False, load=False, file_name='data/target_file.dat',
                 lazy=False):
        # inputs
        self.list_of_variables_to_save = list_of_variables_to_save
        self.save = save
        self.load = load
        self.file_name = file_name
        self.lazy = lazy

        # outputs
        self.start_time_save = None
        self.save_is_successful = None
        self.end_time_save = None
        self.duration_time_save = None

        self.start_time_load = None
        self.load_is_successful = None
        self.end_time_load = None
        self.duration_time_load = None

        self.list_of_loaded_variables = None

        if not self.lazy:
            if self.save:
                self.start_time_save = time.time()
                self.save_is_successful = self.do_save(self.file_name, self.list_of_variables_to_save)
                self.end_time_save = time.time()
                self.duration_time_save = self.end_time_save - self.start_time_save

            if self.load:
                self.start_time_load = time.time()
                self.load_is_successful, self.list_of_loaded_variables = self.do_load(self.file_name)
                self.end_time_load = time.time()
                self.duration_time_load = self.end_time_load - self.start_time_load

    def do_save(self, file_name, list_of_variables_to_save):
        parent_path = Path(file_name).parent
        parent_path.mkdir(parents=True, exist_ok=True)  # create the parent folder; note that this does not overwrite the existing parent folder
        with open(file_name, "wb") as f:
            pickle.dump(list_of_variables_to_save, f)
        return True  # success

    def do_load(self, file_name):
        if path.exists(file_name):
            with open(file_name, "rb") as f:
                list_of_loaded_variables = pickle.load(f)
            return True, list_of_loaded_variables
        else:
            return False, None
