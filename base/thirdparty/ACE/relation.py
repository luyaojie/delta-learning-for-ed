#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/26
import sys
from .relation_mention import RelationMention
from .type_config import pre_defined_time_role


class Relation(object):

    def __init__(self, id, type, subtype, modality, tense, arg1, arg2):
        self.id = id
        self.type = type
        self.subtype = subtype
        self.modality = modality
        self.tense = tense
        self.arg1 = arg1
        self.arg2 = arg2
        self.mentions = list()
        self.timeRoles = None

    @staticmethod
    def from_xml_tags(tags, document, docid=None):
        arg1 = None
        arg2 = None
        time_roles = dict()
        argument_tags = tags.find_all('relation_argument')
        for argument_tag in argument_tags:
            role = argument_tag.get('ROLE')
            if role == 'Arg-1':
                arg1 = document.find_entity(argument_tag.get('REFID'))
            elif role == 'Arg-2':
                arg2 = document.find_entity(argument_tag.get('REFID'))
            elif role in pre_defined_time_role:
                time_roles[role] = document.find_timex(argument_tag.get('REFID'))
            else:
                sys.stderr.write("Invalid ROLE: [%s] for relation" % argument_tag.get('ROLE'))

        relation = Relation(id=tags.get('ID'),
                            type=tags.get('TYPE'),
                            subtype=tags.get('SUBTYPE'),
                            modality=tags.get('MODALITY'),
                            tense=tags.get('TENSE'),
                            arg1=arg1,
                            arg2=arg2)
        if len(time_roles) > 0:
            relation.timeRoles = time_roles

        # add relation mentions
        relation.mentions += [RelationMention.from_xml_tags(rm, document, docid=docid, relation=relation)
                              for rm in tags.find_all("relation_mention")]

        return relation

    def to_xml_tags(self):
        xml_tags = list()
        relation_head = '<relation ID="%s" TYPE="%s" ' % (self.id, self.type)
        if self.subtype is not None and self.subtype != '':
            relation_head += ' SUBTYPE="%s"' % self.subtype
        if self.modality is not None and self.modality != '':
            relation_head += ' MODALITY="%s"' % self.modality
        if self.tense is not None and self.tense != '':
            relation_head += ' TENSE="%s"' % self.tense
        relation_head += '>'

        xml_tags += [relation_head]
        xml_tags += ['<relation_argument REFID="%s" ROLE="Arg-1"/>' % self.arg1.id]
        xml_tags += ['<relation_argument REFID="%s" ROLE="Arg-2"/>' % self.arg2.id]
        xml_tags += [rm.to_xml_tags() for rm in self.mentions]

        xml_tags += ['</relation>']

        return '\n'.join(xml_tags)

    def find_mention(self, m_id):
        for m in self.mentions:
            if m.id == m_id:
                return m
        return None

    def get_label(self):
        label = self.type
        if self.subtype is not None and self.subtype != '':
            label += ':%s' % self.subtype
        return label
