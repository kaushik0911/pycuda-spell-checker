# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 06:08:50 2016

@author: Rexxar
"""
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from collections import Counter
import time

import numpy
import sys

pycuda.tools.clear_context_caches()

word = 'subset'
letters = 'abcdefghijklmnopqrstuvwxyz'

split_set = [(word[:letter], word[letter:]) for letter in range(len(word) + 1)]
subset_of_delete = [leftside_set + rightside_set[1:] for leftside_set, rightside_set in split_set if rightside_set]
transposes = [leftside_set + rightside_set[1] + rightside_set[0] + rightside_set[2:] for leftside_set, rightside_set in split_set if len(rightside_set)>1]
replaces   = [leftside_set + char + rightside_set[1:] for leftside_set, rightside_set in split_set if rightside_set for char in letters]
inserts    = [leftside_set + char + rightside_set for leftside_set, rightside_set in split_set for char in letters]

all_sets = list(set(subset_of_delete + transposes + replaces + inserts))

def search(input_word):

    data = numpy.array(list(
    ' subset sbset ubset asdasd suset subset sbset ubset asdasd suset ubset asdasd suset subset sbset ubset asdasd suset subset sbset ubset asdasd suset subset sbset ubset asdasd suset subset sbset ubset asdasd suset subset sbset ubset asdasd suset subset sbset ubset asdasd suset subset sbset ubset asdasd suset subset sbset ubset asdasd suset subset sbset ubset asdasd suset subset sbset ubset asdasd suset subset sbset ubset asdasd suset subset sbset ubset asdasd suset subset sbset ubset asdasd suset '))
    word = numpy.array(list(input_word)) 
    result = numpy.zeros(shape=(1,len(data)));
    
    word = word.astype(numpy.str_)
    result = result.astype(numpy.int)
    data = data.astype(numpy.str_)
    
    test_data_gpu = cuda.mem_alloc(sys.getsizeof(data))
    word_gpu = cuda.mem_alloc(sys.getsizeof(word))
    result_gpu = cuda.mem_alloc(sys.getsizeof(result))
    
    cuda.memcpy_htod(test_data_gpu,data)
    cuda.memcpy_htod(word_gpu,word)
    
    mod = SourceModule("""
    
    __global__ void searchKeywordKernel(int *result, char *data, char *keyword, int size)
    {
       int i = blockIdx.x * blockDim.x + threadIdx.x;     
       int value = 0;
       
       if(data[i-1] == ' ' && (data[i] == keyword[0])){
        
            for(int k = 1; k < size ; k++ ){
        
                if(data[i + k] != keyword[k]){
                    value = 0;
                    break;
                }
                value = 1;
            }
    
            if (value == 1 && data[i+size] == ' '){ result[i] = 1; }
        }
    
    }
    
    """)
    
    func = mod.get_function("searchKeywordKernel")
    
    
    func(result_gpu,test_data_gpu,word_gpu,numpy.int32(len(word)),block=(len(data),1,1), grid=(1,1,1))
    
    cuda.memcpy_dtoh(result, result_gpu)
    
#    print result
    
    total_matches = 0;
    
    for i in range(len(data)):
        if result[0][i] == 1:
            total_matches += 1

    if total_matches >= 1:    
        print word , total_matches
    
    pycuda.tools.clear_context_caches()

ran = 0
start = time.time()

for i in range(len(all_sets)):
    search(all_sets[i])

#search('subset')

end = time.time()
ran += (end - start)
print "Total run time = ", ran