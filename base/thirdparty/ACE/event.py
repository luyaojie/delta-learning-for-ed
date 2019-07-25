#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/27
from .event_argument import EventArgument
from .event_mention import EventMention


class Event(object):

    def __init__(self, id, type, subtype):
        """
        :param id:       str
        :param type:     str
        :param subtype:  str
        """
        self.id = id
        self.type = type
        self.subtype = subtype
        self.modality = "Asserted"
        self.polarity = "Positive"
        self.genericity = "Specific"
        self.tense = "Past"

        self.arguments = list()
        self.mentions = list()

    @staticmethod
    def from_xml_tags(tags, document, doc_id=None):
        event = Event(id=tags.attrs['ID'],
                      type=tags.attrs['TYPE'],
                      subtype=tags.attrs['SUBTYPE'],
                      )
        event.modality = tags.attrs['MODALITY']
        event.polarity = tags.attrs['POLARITY']
        event.genericity = tags.attrs['GENERICITY']
        event.tense = tags.attrs["TENSE"]

        # add event arguments
        event.arguments += [EventArgument.from_xml_tags(ea, document, doc_id)
                            for ea in tags.find_all('event_argument')]

        # add event mention
        event.mentions += [EventMention.from_xml_tags(evm, document, event, doc_id)
                           for evm in tags.find_all('event_mention')]

        return event

    def to_xml_tags(self):
        xml_tags = list()

        event_head = '<event ID="%s" TYPE="%s" SUBTYPE="%s"' % (self.id, self.type, self.subtype)
        if self.modality is not None and self.modality != '':
            event_head += ' MODALITY="%s"' % self.modality
        if self.polarity is not None and self.polarity != '':
            event_head += ' POLARITY="%s"' % self.polarity
        if self.genericity is not None and self.genericity != '':
            event_head += ' GENERICITY="%s"' % self.genericity
        if self.tense is not None and self.tense != '':
            event_head += ' TENSE="%s"' % self.tense
        event_head += '>'
        xml_tags.append(event_head)

        if len(self.arguments) > 0:
            event_arguments = '\n'.join([eva.to_xml_tags() for eva in self.arguments])
            xml_tags.append(event_arguments)

        if len(self.mentions) > 0:
            event_mentions = '\n'.join([evm.to_xml_tags() for evm in self.mentions])
            xml_tags.append(event_mentions)

        event_end = '</event>'
        xml_tags.append(event_end)

        return '\n'.join(xml_tags)

    def get_label(self):
        label = self.type
        if self.subtype is not None and self.subtype != '':
            label += ':%s' % self.subtype
        return label
