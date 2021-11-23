import time
import datetime
class TimerDefault():
    def __init__(self, time_unit='minute'):
        # inputs
        self.time_unit = time_unit
        # outputs
        self.time_start = None
        self.time_end = None
        self.time_total_seconds = None
        self.time_duration = None

    def start(self):
        self.time_start = datetime.datetime.now()

    def end(self):
        self.time_end = datetime.datetime.now()

    def get_duration(self):
        self.total_seconds = (self.time_end - self.time_start).total_seconds()
        if self.time_unit == 'minute':
            self.time_duration = self.total_seconds / 60.0
        elif self.time_unit == 'hour':
            self.time_duration = self.total_seconds / 60.0 / 60.0
        else:
            raise NotImplementedError
        return self.time_duration
