#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/19
from .mention import Mention
from .charseq import CharSeq


class EntityName(Mention):
    """
    the extent of the mention, with start and end positions based on
    ACE offsets (excluding XML tags).
    """

    def __init__(self, extent):
        """
        :param extent: charseq
        """
        Mention.__init__(self, id, extent, extent)

    @staticmethod
    def from_xml_tags(tags, docid=None):
        charseqs = tags.find_all("charseq")
        extent = CharSeq.from_xml_tags(charseqs[0], docid)
        return EntityName(extent)

    def to_xml_tags(self):
        return '    <name NAME="%s">\n      %s\n    </name>' % (self.extent.text, self.extent.to_xml_tags())
