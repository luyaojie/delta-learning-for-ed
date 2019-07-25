#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger

def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


def write_str_to_out(out, write_str):
    out.write(any2utf8(write_str))


def write_list_to_out(out, write_list, split_symbol='\t'):
    new_list = [str(ele)
                if not isinstance(ele, unicode) and not isinstance(ele, str)
                else ele
                for ele in write_list]
    write_str_to_out(out, split_symbol.join(new_list) + '\n')
