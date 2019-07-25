#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/19


class Mention:

    def __init__(self, id, extent, head):
        """
        :param id:      str
        :param extent:  CharSeq
        :param head:    CharSeq
        """
        self.id = id
        self.extent = extent
        self.head = head
