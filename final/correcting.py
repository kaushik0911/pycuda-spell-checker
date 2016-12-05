# -*- coding: utf-8 -*-

import main
import sequential
import time
import non_stream

#sliit is accredited as a degree awarding institute under the universities act of sri lanka

text = 'slit is accredited as digree awarding institute under the univercities act of sri lanka'
word_list = text.split()

after_correction = []

time_p = 0.0000

print '\n------- without CUDA streams -------'

for word in word_list:
    return_value = non_stream.final(word)    
    print return_value[0]
    after_correction.append(return_value[0])
    time_p += return_value[1]
    
print '\ninput  : ', text
print '\noutput : ',' '.join(after_correction)   
print '\n\ntotal time for parallel ( without streams ) : ',time_p

after_correction = []

time_p = 0.0000

print '\n------- with CUDA streams -------'

for word in word_list:
    return_value = main.search(word)    
    print return_value[0]
    after_correction.append(return_value[0])
    time_p += return_value[1]
    
print '\ninput  : ', text
print '\noutput : ',' '.join(after_correction)   
print '\n\ntotal time for parallel ( with streams ) : ',time_p

print '\n------- sequential program -------'

after_correction = []

ran = 0
start = time.time()

for word in word_list:
    return_value = sequential.correction(word)   
    after_correction.append(return_value)

end = time.time()
ran += (end - start)

print '\noutput : ',' '.join(after_correction)   
print '\n\ntotal time for sequential : ',ran
