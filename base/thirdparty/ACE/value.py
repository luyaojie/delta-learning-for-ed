#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/26
from .value_mention import ValueMention


class Value(object):

    def __init__(self, id, type, subtype):
        """
        :param id:       str
        :param type:     str
        :param subtype:  str
        :param entClass: str
        """
        self.id = id
        self.type = type
        if subtype is None:
            subtype = ""
        self.subtype = subtype
        self.mentions = list()

    @staticmethod
    def from_xml_tags(tags, docid=None):
        value = Value(id=tags.get('ID'),
                      type=tags.get('TYPE'),
                      subtype=tags.get('SUBTYPE')
                      )
        # add value mentions
        value.mentions += [ValueMention.from_xml_tags(vm, docid=docid, value=value) for vm in
                           tags.find_all("value_mention")]
        return value

    def to_xml_tags(self):
        xml_tags = list()

        value_head = '<value ID="%s" TYPE="%s"' % (self.id, self.type)
        if self.subtype is not None and self.subtype != "":
            value_head += ' SUBTYPE="%s"' % self.subtype
        value_head += '>'

        xml_tags += [value_head]
        xml_tags += [vm.to_xml_tags() for vm in self.mentions]
        xml_tags += ['</value>']
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
