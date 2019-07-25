#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/26
from .mention import Mention
from .charseq import CharSeq


class ValueMention(Mention):

    def __init__(self, id, value, extent):
        """
        :param id:     str
        :param value:  Value
        :param extent: charseq
        """
        Mention.__init__(self, id, extent, extent)
        self.value = value

    @staticmethod
    def from_xml_tags(tags, docid=None, value=None):
        vm_id = tags.get('ID')
        extent = CharSeq.from_xml_tags(tags.find_all('extent')[0].find_all('charseq')[0], docid=docid)
        return ValueMention(id=vm_id, value=value, extent=extent)

    def to_xml_tags(self):
        vm_head = '  <value_mention ID="%s"">' % self.id
        extent = '    <extent>\n      %s\n    </extent>' % self.extent.to_xml_tags()
        vm_end = '  </value_mention>'
        return '\n'.join([vm_head, extent, vm_end])

    def get_label(self):
        return self.value.get_label()
