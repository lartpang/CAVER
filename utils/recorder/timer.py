# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import functools
from datetime import datetime

from utils.py_utils import construct_print


def CustomizedTimer(cus_msg):
    def Timer(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            construct_print(f"{cus_msg} start: {start_time}")
            results = func(*args, **kwargs)
            construct_print(f"the time of {cus_msg}: {datetime.now() - start_time}")
            return results

        return wrapper

    return Timer


class TimeRecoder:
    def __init__(self):
        self._start_time = datetime.now()
        self._has_start = False

    def start(self, msg=""):
        self._start_time = datetime.now()
        self._has_start = True
        if msg:
            construct_print(f"[{self._start_time}] {msg}")

    def now_and_reset(self, pre_msg=""):
        if not self._has_start:
            raise AttributeError("You must call the `.start` method before the `.now_and_reset`!")
        self._has_start = False
        end_time = datetime.now()
        construct_print(f"[{end_time}] {pre_msg} {end_time - self._start_time}")
        self.start()

    def now(self, pre_msg=""):
        if not self._has_start:
            raise AttributeError("You must call the `.start` method before the `.now`!")
        self._has_start = False
        end_time = datetime.now()
        construct_print(f"[{end_time}] {pre_msg} {end_time - self._start_time}")
