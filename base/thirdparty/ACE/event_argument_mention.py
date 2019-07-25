#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/27
from .charseq import CharSeq
from .mention import Mention


class EventArgumentMention(Mention):

    def __init__(self, extent, value, role):
        Mention.__init__(self, value.id, extent, extent)
        self.extent = extent
        self.value = value
        self.role = role

    @staticmethod
    def from_xml_tags(tags, document, doc_id=None):
        id = tags.get('REFID')
        role = tags.get('ROLE')
        extent = CharSeq.from_xml_tags(tags.find_all('extent')[0].find_all('charseq')[0], doc_id)
        value = document.find_entity_value_timex_mention(id)
        return EventArgumentMention(extent, value, role)

    def to_xml_tags(self):
        evam_head = '    <event_mention_argument REFID="%s" ROLE="%s">' % (self.value.id, self.role)
        extent = '      <extent>\n        %s\n      </extent>' % self.extent.to_xml_tags()
        evam_end = '    </event_mention_argument>'
        return '\n'.join([evam_head, extent, evam_end])

    def get_label(self):
        return self.role
