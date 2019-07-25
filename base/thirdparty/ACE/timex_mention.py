#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/19
from .mention import Mention
from .charseq import CharSeq


class TimexMention(Mention):

    def __init__(self, id, timex, extent):
        """
        :param id:     str
        :param timex:  Timex
        :param extent: charseq
        """
        Mention.__init__(self, id, extent, extent)
        self.timex = timex

    @staticmethod
    def from_xml_tags(tags, docid=None, timex=None):
        tm_id = tags.get('ID')
        extent = CharSeq.from_xml_tags(tags.find_all('extent')[0].find_all('charseq')[0], docid=docid)
        return TimexMention(id=tm_id, timex=timex, extent=extent)

    def to_xml_tags(self):
        vm_head = '  <timex2_mention ID="%s"">' % self.id
        extent = '    <extent>\n      %s\n    </extent>' % self.extent.to_xml_tags()
        vm_end = '  </timex2_mention>'
        return '\n'.join([vm_head, extent, vm_end])

    def get_label(self):
        return self.timex.get_label()
