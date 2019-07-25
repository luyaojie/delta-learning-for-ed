#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/19
from .mention import Mention
from .charseq import CharSeq


class EntityMention(Mention):

    def __init__(self, id, entity, type, extent, head, role):
        """
        :param id:     str
        :param entity: Entity
        :param extent: charseq
        :param head:   charseq
        """
        Mention.__init__(self, id, extent, head)
        self.entity = entity
        # the type of the mention:  NAME, NOMINAL, or PRONOUN.
        self.type = type
        # for entities of type GPE, the role of the mention (ORG, LOC, GPE, or PER).
        self.role = role

    @staticmethod
    def from_xml_tags(tags, docid=None, entity=None):
        em_id = tags.get('ID')
        em_type = tags.get('TYPE')
        em_role = tags.get('ROLE')
        extents = tags.find_all('extent')
        heads = tags.find_all('head')
        extent_charseq = CharSeq.from_xml_tags(extents[0].find_all('charseq')[0], docid)
        head_charseq = CharSeq.from_xml_tags(heads[0].find_all('charseq')[0], docid)
        return EntityMention(id=em_id, entity=entity, type=em_type,
                             extent=extent_charseq, head=head_charseq,
                             role=em_role)

    def to_xml_tags(self):
        em_head = '  <entity_mention ID="%s" TYPE="%s" ROLE="%s">' % (self.id, self.type, self.role)
        extent = '    <extent>\n      %s\n    </extent>' % self.extent.to_xml_tags()
        head = '    <head>\n      %s\n    </head>' % self.head.to_xml_tags()
        em_end = '  </entity_mention>'
        return '\n'.join([em_head, extent, head, em_end])

    def get_label(self):
        return self.entity.get_label()
