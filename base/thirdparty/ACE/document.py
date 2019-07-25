#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/26
import sys

from .entity import Entity
from .timex import Timex
from .value import Value
from .relation import Relation
from .event import Event


class Document:

    def __init__(self, source_text, source_file, source_type, docid):
        self.source_text = source_text
        self.source_file = source_file
        self.source_type = source_type
        self.docid = docid

        self.entities = list()
        self.time_expressions = list()
        self.values = list()

        self.relations = list()
        self.events = list()

        self.analysis_result = None

    @staticmethod
    def from_apf_document(tags, filetext):
        source_file_node = tags.find_all('source_file')[0]
        source_file = source_file_node.get('URI')
        source_type = source_file_node.get('SOURCE')

        document_node = tags.find_all('document')[0]
        docid = document_node.get('DOCID')
        document = Document(source_text=filetext, source_file=source_file, source_type=source_type, docid=docid)

        entities = [Entity.from_xml_tags(entity_xmltag, docid=docid)
                    for entity_xmltag in document_node.find_all('entity')]
        document.entities += entities

        time_expressions = [Timex.from_xml_tags(timex_xmltag, docid=docid)
                            for timex_xmltag in document_node.find_all('timex2')]
        document.time_expressions += time_expressions

        values = [Value.from_xml_tags(value_xmltag, docid=docid)
                  for value_xmltag in document_node.find_all('value')]
        document.values += values

        relations = [Relation.from_xml_tags(relation_xmltag, document, docid)
                     for relation_xmltag in document_node.find_all('relation')]
        document.relations += relations

        events = [Event.from_xml_tags(event_xmltag, document, docid)
                  for event_xmltag in document_node.find_all('event')]
        document.events += events
        return document

    def to_apf_document(self):
        xml_tags = list()

        xml_head = '<?xml version="1.0"?>'
        source_file_head = '<source_file URI="%s" SOURCE="%s">' % (self.source_file, self.source_type)
        source_file_end = '</source_file>'
        document_head = '<document DOCID="%s">' % self.docid
        document_end = '</document>'

        xml_tags += [xml_head, source_file_head, document_head]

        # add entity
        if len(self.entities) > 0:
            xml_tags += [e.to_xml_tags() for e in self.entities]
        if len(self.time_expressions) > 0:
            xml_tags += [te.to_xml_tags() for te in self.time_expressions]
        if len(self.values) > 0:
            xml_tags += [v.to_xml_tags() for v in self.values]
        if len(self.relations) > 0:
            xml_tags += [rel.to_xml_tags() for rel in self.relations]
        if len(self.events) > 0:
            xml_tags += [ev.to_xml_tags() for ev in self.events]

        xml_tags += [document_end, source_file_end]
        return '\n'.join(xml_tags)

    def find_entity(self, eid):
        for entity in self.entities:
            if entity.id == eid:
                return entity
        sys.stderr.write("Can't find Entity Named %s\n" % eid)
        return None

    def find_entity_mention(self, em_id):
        for entity in self.entities:
            em = entity.find_mention(em_id)
            if em:
                return em
        sys.stderr.write("Can't find Entity Mention Named %s\n" % em_id)
        return None

    def find_entity_mention_by_charseq(self, charseq, exact=True):
        result = list()
        for e in self.entities:
            for em in e.mentions:
                if exact:
                    if em.head.exact_match(charseq, check_doc=True):
                        result.append(em)
                else:
                    if em.head.partial_match(charseq, check_doc=True):
                        result.append(em)
        return result

    def find_timex(self, tid):
        for timex in self.time_expressions:
            if timex.id == tid:
                return timex
        sys.stderr.write("Can't find Time Expression Named %s\n" % tid)
        return None

    def find_timex_mention(self, tm_id):
        for timex in self.time_expressions:
            tm = timex.find_mention(tm_id)
            if tm:
                return tm
        sys.stderr.write("Can't find Timex Mention Named %s\n" % tm_id)
        return None

    def find_timex_mention_by_charseq(self, charseq, exact=True):
        result = list()
        for e in self.time_expressions:
            for em in e.mentions:
                if exact:
                    if em.head.exact_match(charseq, check_doc=True):
                        result.append(em)
                else:
                    if em.head.partial_match(charseq, check_doc=True):
                        result.append(em)
        return result

    def find_value_mention_by_charseq(self, charseq, exact=True):
        result = list()
        for e in self.values:
            for em in e.mentions:
                if exact:
                    if em.head.exact_match(charseq, check_doc=True):
                        result.append(em)
                else:
                    if em.head.partial_match(charseq, check_doc=True):
                        result.append(em)
        return result

    def find_event_mention_nugget_by_charseq(self, charseq, exact=True):
        result = list()
        for event in self.events:
            for evm in event.mentions:
                if exact:
                    if evm.nugget.exact_match(charseq, check_doc=True):
                        result.append(evm)
                else:
                    if evm.nugget.partial_match(charseq, check_doc=True):
                        result.append(evm)
        return result

    def find_entity_value_timex(self, eid):
        for timex in self.time_expressions:
            if timex.id == eid:
                return timex
        for value in self.values:
            if value.id == eid:
                return value
        return self.find_entity(eid)

    def find_entity_value_timex_mention(self, emid):
        for timex in self.time_expressions:
            tm = timex.find_mention(emid)
            if tm:
                return tm
        for value in self.values:
            vm = value.find_mention(emid)
            if vm:
                return vm
        return self.find_entity_mention(emid)
