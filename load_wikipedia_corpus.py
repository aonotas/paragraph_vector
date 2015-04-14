#!/usr/bin/env python
# -*- coding: utf-8 -*-

import MySQLdb
from prettyprint import pp, pp_str
from config import database_config 

def get_wiki_text():
    connect = MySQLdb.connect(**database_config)
    cur = connect.cursor(MySQLdb.cursors.DictCursor)

    sql = "SELECT * FROM page_wakati order by page_id asc;"
    sentences, index2pageid, pageid2title = execute_sql(connect,cur,sql)

    cur.close()
    connect.close()

    return sentences, index2pageid, pageid2title

def execute_sql(connect,cur, sql):

    cur.execute(sql)
    sentences = []
    index2pageid = {}
    pageid2title = {}
    index = 0 
    for row in iter(cur):
        page_id = row["page_id"]
        page_title = row["page_title"]
        page_text = row["page_text"]

        page_title = page_title.decode("utf-8")
        page_text = page_text.decode("utf-8")
        if len(page_text) < 10:
            continue
        index2pageid.update({index:page_id})
        pageid2title.update({page_id:page_title})
        sentences.append(page_text)
        index += 1
    return sentences, index2pageid, pageid2title


if __name__ == '__main__':
    sentences, index2pageid, pageid2title = get_wiki_text()
    pp(type(sentences[0]))
    print len(sentences)
    print len(index2pageid.keys())
    # pp(sentences)