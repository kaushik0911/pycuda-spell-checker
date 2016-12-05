# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 06:08:50 2016

@author: Rexxar
"""
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

import insert
import replace

import numpy
import sys

pycuda.tools.clear_context_caches()

def search(word):
    
    split_set = [(word[:letter], word[letter:]) for letter in range(len(word) + 1)]
    subset_of_delete = [leftside_set + rightside_set[1:] for leftside_set, rightside_set in split_set if rightside_set]
    transposes = [leftside_set + rightside_set[1] + rightside_set[0] + rightside_set[2:] for leftside_set, rightside_set in split_set if len(rightside_set)>1]
    replaces   = replace.replace(word)
    inserts    = insert.insert(word)
    
    all_sets = list(set(subset_of_delete + transposes + replaces + inserts))
    
    first_if = len(subset_of_delete[0])
    second_if = len(transposes[0])
    third_if = len(inserts[0])

    data = numpy.array(list(open('text_doc.txt').read()))
    
    number_of_words =  len(all_sets) 
    

    cpu_data_word, gpu_data_word = [], []
    
    for i in range(number_of_words):
        cpu_data_word.append(numpy.array(list(all_sets[i])).astype(numpy.str_))
        gpu_data_word.append(cuda.mem_alloc(sys.getsizeof(cpu_data_word[i])))


    data = data.astype(numpy.str_)
    test_data_gpu = cuda.mem_alloc(sys.getsizeof(data))
    cuda.memcpy_htod(test_data_gpu,data)

    cpu_result_test_data_set, result_data_word = [], []

    for i in range(number_of_words):

        cpu_result_test_data_set.append(numpy.zeros(shape=(1,len(data))).astype(numpy.int))
        result_data_word.append(cuda.mem_alloc(sys.getsizeof(cpu_result_test_data_set[i])))

    for k in range(number_of_words):
        
        cuda.memcpy_htod(gpu_data_word[k], cpu_data_word[k])
        cuda.memcpy_htod(result_data_word[k], cpu_result_test_data_set[k])  

    stream = []
    
    for i in range(number_of_words):
        
        stream.append(cuda.Stream())

   
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
    
    ran = 0
    start = time.time()

    for k in range(number_of_words):
        
        if( len(cpu_data_word[k]) == first_if):
            func( result_data_word[k] , test_data_gpu , gpu_data_word[k] , numpy.int32(first_if) , block=(512,1,1) , grid=(len(data)/512,1,1), stream = stream[k])
        
        elif( len(cpu_data_word[k]) == second_if):
            func( result_data_word[k] , test_data_gpu , gpu_data_word[k] , numpy.int32(second_if) , block=(512,1,1) , grid=(len(data)/512,1,1), stream = stream[k])
        
        elif( len(cpu_data_word[k]) == third_if):
            func( result_data_word[k] , test_data_gpu , gpu_data_word[k] , numpy.int32(third_if) , block=(512,1,1) , grid=(len(data)/512,1,1), stream = stream[k])
           
    cuda.Stream().synchronize()           
           
    end = time.time()
    ran += (end - start)
    print "run time = ", ran
    
    for k in range(number_of_words):

        cuda.memcpy_dtoh(cpu_result_test_data_set[k], result_data_word[k])
  
    pycuda.tools.clear_context_caches()


    word_max = None
    max_value = 0
    
    for k in range(number_of_words):  
        result = cpu_result_test_data_set[k]       
     
        total_matches = 0;
    
        for i in range(len(data)):
            if result[0][i] == 1:
                total_matches += 1
    
        if total_matches >= 1 and total_matches > max_value:
            
            max_value = total_matches            
            word_max = cpu_data_word[k]
    
    if word_max is None:
        return word, ran       

    return ''.join(word_max) , ran  
        
        
    