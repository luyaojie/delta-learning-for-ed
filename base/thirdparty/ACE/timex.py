#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/19
from .timex_mention import TimexMention


class Timex(object):

    def __init__(self, id, val, anchorVal, anchorDir, timexSet, timexMod):
        """
        an Ace Timex2 time expression.  The 'id' field is inherited from its
        superclass, AceEventArgumentValue.
        """
        self.id = id
        self.val = val
        self.anchorVal = anchorVal
        self.anchorDir = anchorDir
        self.set = timexSet
        self.mod = timexMod
        self.mentions = list()

    @staticmethod
    def from_xml_tags(tags, docid=None):
        timex = Timex(id=tags.get('ID'),
                      val=tags.get('VAL'),
                      anchorVal=tags.get('ANCHOR_VAL'),
                      anchorDir=tags.get('ANCHOR_DIR'),
                      timexSet=tags.get('SET'),
                      timexMod=tags.get('MOD'),
                      )
        # add value mentions
        timex.mentions += [TimexMention.from_xml_tags(tm, docid=docid, timex=timex) for tm in
                           tags.find_all("timex2_mention")]
        return timex

    def to_xml_tags(self):
        xml_tags = list()
        timex_head = '<timex2 ID="%s"' % self.id
        if self.val is not None and self.val != '':
            timex_head += ' VAL="%s"' % self.val
        if self.anchorVal is not None and self.anchorVal != '':
            timex_head += ' ANCHOR_VAL="%s"' % self.anchorVal
        if self.anchorDir is not None and self.anchorDir != '':
            timex_head += ' ANCHOR_DIR="%s"' % self.anchorDir
        if self.set is not None and self.set != '':
            timex_head += ' SET="%s"' % self.set
        if self.mod is not None and self.mod != '':
            timex_head += ' MOD="%s"' % self.mod
        timex_head += ">"
        xml_tags += [timex_head]
        xml_tags += [tm.to_xml_tags() for tm in self.mentions]
        xml_tags += ['</timex2>']
        return '\n'.join(xml_tags)

    def get_label(self):
        if self.val is not None and self.val.startswith('P') and self.val.startswith('Y'):
            return 'Age'
        else:
            return 'Time'

    def find_mention(self, m_id):
        for m in self.mentions:
            if m.id == m_id:
                return m
        return None
