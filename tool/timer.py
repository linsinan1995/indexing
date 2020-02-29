'''
Filename: e:\indexing\tool\timer.py
Path: e:\indexing\tool
Created Date: Sunday, February 23rd 2020, 1:10:23 am
Author: linsinan1995

Copyright (c) 2020 Lin Sinan
'''

from datetime import datetime


class timer:
    __start = None
    __unit = None
    __history = []

    @staticmethod
    def start(unit = "second"):
        t =  datetime.now()
        timer.set_properties(t, unit)
    
    @staticmethod
    def end():
        end = datetime.now()
        start = timer.get_start()
        unit = timer.get_unit()

        assert(start != None)
        
        tt = end - start
        if unit in ["ms", "millisecond", 1]:
            unit = "ms" if isinstance(unit, int) else unit
            tt = (tt.days * 24 * 60 * 60 + tt.seconds) * 10e3 + tt.microseconds * 10e-3
        elif unit in ["us", "microsecond","μs", 2]:
            unit = "us" if isinstance(unit, int) else unit
            tt = (tt.days * 24 * 60 * 60 + tt.seconds) * 10e6 + tt.microseconds
        else:
            unit = "second" if unit != "s" else "s" 
            tt = tt.total_seconds()

        print("TIME: {:.2f} {}".format(tt, unit))
        timer.set_properties(None, None)
        timer.record(tt)

    @classmethod
    def get_start(cls):
        return cls.__start

    @classmethod
    def get_unit(cls):
        return cls.__unit

    @classmethod
    def set_properties(cls, start_time, unit = None):
        cls.__start = start_time
        # if unit is None:
        #     cls.__unit = unit
        if not unit is None:
            cls.__unit = unit
            
    @classmethod
    def record(cls, delta_time):
        cls.__history.append(delta_time)

    @classmethod
    def get_record(cls):
        return cls.__history

if __name__ == "__main__":
    timer.start()
    b = []
    for i in range(7000**2):
        b.append(i)
    timer.end()

    timer.start(1)
    b = []
    for i in range(7000**2):
        b.append(i)
    timer.end()

    timer.start(2)
    b = []
    for i in range(7000**2):
        b.append(i)
    timer.end()

    timer.start(1)
    b = []
    for i in range(1000**2):
        b.append(i)

    timer.end()

    timer.start(2)
    b = []
    for i in range(1000**2):
        b.append(i)
    timer.end()

    print(timer.get_record())