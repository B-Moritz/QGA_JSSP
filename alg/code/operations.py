class Operation:
    def __init__(self, job, machine, duration, start):
        self.job = job
        self.machine = machine
        self.duration = duration
        self.start = start


    def get_completion_time(self):
        return self.start + self.duration

    def __str__(self):
        return f"Operation_{self.job}_{self.machine}"