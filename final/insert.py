# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:10:57 2016

@author: Rexxar
"""

import pycuda.driver as cuda
import pycuda.autoinit , time
from pycuda.compiler import SourceModule

import numpy
import sys

def insert(word):

    letters = 'abcdefghijklmnopqrstuvwxyz'
    word_char_arry = (pow(len(word)+1,2) * len(letters))
    thread_size = 26
    
    empty_chars = ['1' for x in range(word_char_arry)]
    a = numpy.char.array(empty_chars)
    b = numpy.char.array(list(letters))
    c = numpy.char.array(list(word))

    a = a.astype(numpy.character)
    b = b.astype(numpy.character)
    c = c.astype(numpy.character)
    
    a_gpu = cuda.mem_alloc(sys.getsizeof(a))
    b_gpu = cuda.mem_alloc(sys.getsizeof(b))
    c_gpu = cuda.mem_alloc(sys.getsizeof(c))
    
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)
    cuda.memcpy_htod(c_gpu, c)
    
    mod = SourceModule("""
      __global__ void doublify(char *a,char *b,char *word, int size)
      {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        char al = b[idx];
        int count = 0;
    
        int cal_size = size + 1;
        int mul = cal_size * cal_size;
        
        for(int j=0;j<cal_size;j++){
            
            for(int k=0;k<cal_size;k++){
                    
                if(k==j){
                
                    a[(mul*idx)+(j*cal_size)+k] = al;
                }  
                
                else{
    
                    a[(mul*idx)+(j*cal_size)+k] =  word[count];
                    count++;
                }
            }
            
            count = 0;
        }
        
        count = 0;
      }
      """)
      
    func = mod.get_function("doublify")
    
#    ran = 0.0000
#    start = time.time() 
      
    func(a_gpu, b_gpu, c_gpu, numpy.int32(len(word)), block=(thread_size,1,1),grid = (1, 1))
    
#    end = time.time()
#    ran += (end - start)
#    print "Total run time to insert = ", ran
      
    cuda.memcpy_dtoh(a, a_gpu)
    
    n = len(word) + 1
    
    pycuda.tools.clear_context_caches()
    a_gpu.free()
    b_gpu.free()
    c_gpu.free()
    
    return [''.join(str(e) for e in a)[i:i+n] for i in range(0, len(a), n)]