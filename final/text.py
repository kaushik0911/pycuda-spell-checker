# -*- coding: utf-8 -*-

import re

def words(text): return re.findall(r'\w+', text.lower())
    
all_data = words(open('text_file.txt').read())

fo = open("text_doc.txt", "wb") 
fo.write(' '.join(all_data))

fo.close()
