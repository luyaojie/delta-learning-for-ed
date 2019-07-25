#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/19
import sys


class CharSeq(object):

    def __init__(self, start, end, text, docid=None, init_use_end=False):
        self.start = start

        # ACE End is End Char Offset, use_end is use in python
        # text = doc[start:use_end]
        if init_use_end:
            self.end = end - 1
            self.use_end = end
        else:
            self.end = end
            self.use_end = end + 1

        self.length = self.end - self.start + 1
        self.docid = docid
        self.text = text

    def __str__(self):
        return "[%s-%s]" % (self.start, self.end)

    @staticmethod
    def from_xml_tags(tags, docid=None):
        """
        <charseq START="153" END="163">Philippines</charseq>
        :param tags: Xml Tags
        :param docid: str
        :return:
        """
        start = int(tags.attrs['START'])
        end = int(tags.attrs['END'])
        text = tags.text
        return CharSeq(start=start, end=end, text=text, docid=docid)

    @staticmethod
    def from_stanford_token(token, docid=None):
        return CharSeq(start=token['characterOffsetBegin'],
                       end=token['characterOffsetEnd'],
                       text=token['originalText'],
                       docid=docid,
                       init_use_end=True)

    def to_xml_tags(self):
        return '<charseq START="%s" END="%s">%s</charseq>' % (self.start, self.end, self.text)

    def exact_match(self, other, check_doc=True, new_line_replace=''):
        """
        :param other: CharSeq
        :param check_doc: Boolean
        :param new_line_replace: str
        :return:
        """
        if check_doc:
            assert self.docid == other.docid
        if self.start == other.start and self.end == other.end:
            # Some word in Chinese has new line
            # 座谈会
            # 座
            # 谈会
            if self.text != other.text.replace('\n', new_line_replace):
                sys.stderr.write("%s\t%s\n" % (self.text, other.text))
            return True
        return False

    def partial_match(self, other, check_doc=True):
        """
        :param other: CharSeq
        :param check_doc: Boolean
        :return:
        """
        if check_doc:
            assert self.docid == other.docid
        if self.start < other.start and self.end < other.start:
            return False
        elif other.start < self.start and other.end < self.start:
            return False
        else:
            return True
