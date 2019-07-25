#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/19
from .entity_mention import EntityMention
from .entity_name import EntityName


class Entity(object):

    def __init__(self, id, type, subtype, entClass):
        """
        :param id:       str
        :param type:     str
        :param subtype:  str
        :param entClass: str
        """
        self.id = id
        self.type = type
        self.subtype = subtype
        self.entClass = entClass
        self.mentions = list()
        self.names = list()

    @staticmethod
    def from_xml_tags(tags, docid=None):
        entity = Entity(id=tags.attrs['ID'],
                        type=tags.attrs['TYPE'],
                        subtype=tags.attrs['SUBTYPE'],
                        entClass=tags.attrs['CLASS'],
                        )
        # add entity mentions
        entity.mentions += [EntityMention.from_xml_tags(em, docid=docid, entity=entity) for em in
                            tags.find_all("entity_mention")]
        # add entity names in attribute
        entity.names += [EntityName.from_xml_tags(ename, docid) for ename in tags.find_all("name")]
        return entity

    def to_xml_tags(self):
        xml_tags = list()
        entity_head = '<entity ID="%s" TYPE="%s" SUBTYPE="%s" CLASS="%s">' % (self.id, self.type,
                                                                              self.subtype, self.entClass)
        xml_tags += [entity_head]
        xml_tags += [em.to_xml_tags() for em in self.mentions]
        if len(self.names) > 0:
            xml_tags += ["  <entity_attributes>"]
            xml_tags += [ename.to_xml_tags() for ename in self.names]
            xml_tags += ["  </entity_attributes>"]
        xml_tags += ['</entity>']
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
