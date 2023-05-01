# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang


class MsgLogger(object):
    def __init__(self, **kwargs):
        for name, path in kwargs.items():
            setattr(self, name, path)

    def __call__(self, name, msg, show=True):
        self._file_logger(name, msg)
        if show:
            self._term_logger(msg)

    def _file_logger(self, name, msg):
        assert hasattr(self, name)
        with open(getattr(self, name), "a") as logger:
            logger.write(f"{msg}\n")

    @staticmethod
    def _term_logger(msg):
        print(msg)
