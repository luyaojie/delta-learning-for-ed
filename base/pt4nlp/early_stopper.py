#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger
import numpy as np


class EarlyStopper:
    def __init__(self,
                 # max_better_num=0, min_better_num=0,
                 min_times=10, min_valid_times=3):
        """
        :return:
        """
        self.min_times = min_times
        self.min_valid_times = min_valid_times

    def check_early_stop(self, max_better_list, min_better_list):
        """
        :param max_better_list: List(list)
        :param min_better_list:  List(list)
        :return:
        """
        min_times = self.min_times
        min_valid_times = self.min_valid_times

        # Check high is better
        for result in max_better_list:
            if len(result) < min_times or len(result) < 2 * min_valid_times:
                return False

            last_result = np.mean(result[len(result) - 2 * min_valid_times:len(result) - min_valid_times])
            curr_result = np.mean(result[len(result) - min_valid_times:])
            # cur is bigger, no stop
            if curr_result > last_result:
                return False

        # Check low is better
        for result in min_better_list:
            if len(result) < min_times or len(result) < 2 * min_valid_times:
                return False

            last_result = np.mean(result[len(result) - 2 * min_valid_times:len(result) - min_valid_times])
            curr_result = np.mean(result[len(result) - min_valid_times:])
            # cur is smaller, no stop
            if curr_result < last_result:
                return False

        return True
