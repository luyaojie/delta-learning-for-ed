#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/27
from .charseq import CharSeq
from .mention import Mention
from .event_argument_mention import EventArgumentMention


class EventMention(Mention):

    def __init__(self, id, nugget, extent, event):
        """
        :param nugget: CharSeq
        """
        Mention.__init__(self, id, extent=extent, head=nugget)
        self.nugget = nugget
        self.argument_mentions = list()
        self.event = event

    @staticmethod
    def from_xml_tags(tags, document, event, doc_id=None):
        evm_id = tags.get('ID')
        nugget = CharSeq.from_xml_tags(tags.find_all('anchor')[0].find_all('charseq')[0], doc_id)

        extent = CharSeq.from_xml_tags(tags.find_all('extent')[0].find_all('charseq')[0], doc_id)

        evm = EventMention(id=evm_id,
                           nugget=nugget,
                           extent=extent,
                           event=event)
        argument_mentions = [EventArgumentMention.from_xml_tags(evam_tags, document, doc_id)
                             for evam_tags in tags.find_all('event_mention_argument')]
        evm.argument_mentions += argument_mentions
        return evm

    def to_xml_tags(self):
        evm_head = '  <event_mention ID="%s">' % self.id
        evm_extent = '    <extent>\n      %s\n    </extent>' % self.extent.to_xml_tags()
        evm_nugget = '    <anchor>\n      %s\n    </anchor>' % self.nugget.to_xml_tags()
        evm_arg_mentions = '\n'.join([evam.to_xml_tags() for evam in self.argument_mentions])
        evm_end = '  </event_mention>'
        return '\n'.join([evm_head, evm_extent, evm_nugget, evm_arg_mentions, evm_end])

    def get_label(self):
        return self.event.get_label()
