#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/27


class EventArgument:

    def __init__(self, role, value):
        self.role = role
        self.value = value

    @staticmethod
    def from_xml_tags(tags, document, docid=None):
        refid = tags.get('REFID')
        role = tags.get('ROLE')
        value = document.find_entity_value_timex(refid)
        ea = EventArgument(role=role, value=value)
        return ea

    def to_xml_tags(self):
        return '  <event_argument REFID="%s" ROLE="%s"/>' % (self.value.id, self.role)
