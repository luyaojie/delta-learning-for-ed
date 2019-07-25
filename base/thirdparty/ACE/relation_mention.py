#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/19
import sys
from .charseq import CharSeq
from .mention import Mention
from .type_config import pre_defined_time_role


class RelationMention(Mention):

    def __init__(self, id, extent, arg1, arg2, relation):
        """
        :param id:     str
        :param extent: charseq
        :param arg1:   Entity
        :param arg2:   Entity
        :param relation: Relation
        """
        Mention.__init__(self, id, extent, extent)
        self.arg1 = arg1
        self.arg2 = arg2
        self.extent = extent
        self.relation = relation
        self.timeRoles = None

    @staticmethod
    def from_xml_tags(tags, document, docid=None, relation=None):
        arg1 = None
        arg2 = None
        time_roles = dict()
        rm_id = tags.get('ID')
        argument_tags = tags.find_all('relation_mention_argument')

        for argument_tag in argument_tags:
            role = argument_tag.get('ROLE')
            if role == 'Arg-1':
                arg1 = document.find_entity_mention(argument_tag.get('REFID'))
            elif role == 'Arg-2':
                arg2 = document.find_entity_mention(argument_tag.get('REFID'))
            elif role in pre_defined_time_role:
                time_roles[role] = document.find_timex_mention(argument_tag.get('REFID'))
            else:
                sys.stderr.write("Invalid ROLE: [%s] for relation\n" % argument_tag.get('ROLE'))

        extent_charseq = CharSeq.from_xml_tags(tags.find_all('extent')[0].find_all('charseq')[0], docid)
        rm = RelationMention(id=rm_id, extent=extent_charseq, arg1=arg1, arg2=arg2, relation=relation)
        if len(time_roles) > 0:
            rm.timeRoles = time_roles
        return rm

    def to_xml_tags(self):
        rm_head = '  <relation_mention ID="%s"">' % self.id
        extent = '    <extent>\n      %s\n    </extent>' % self.extent.to_xml_tags()
        arg1 = self.mention_arg_to_xml_tags('Arg-1', self.arg1)
        arg2 = self.mention_arg_to_xml_tags('Arg-2', self.arg1)
        rm_end = '  </relation_mention>'
        return '\n'.join([rm_head, extent, arg1, arg2, rm_end])

    @staticmethod
    def mention_arg_to_xml_tags(arg_name, arg):
        head = '    <relation_mention_argument REFID="%s" ROLE="%s">' % (arg.id, arg_name)
        body = '      <extent>\n      %s\n    </extent>' % arg.extent.to_xml_tags()
        end = '    </relation_mention_argument>'
        return '\n'.join([head, body, end])
