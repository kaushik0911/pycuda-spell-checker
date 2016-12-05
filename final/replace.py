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

def replace(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    word_char_arry = ( word * len(word) ) * len(letters)
    thread_size = 26
    
    a = numpy.char.array(list(word_char_arry))
    b = numpy.char.array(list(letters))
    
    a = a.astype(numpy.character)
    b = b.astype(numpy.character)
    
    a_gpu = cuda.mem_alloc(sys.getsizeof(a))
    b_gpu = cuda.mem_alloc(sys.getsizeof(b))
    
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)
    
    mod = SourceModule("""
      __global__ void doublify(char *a,char *b, int size)
      {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        char al = b[idx];
        
        int mul = size * size;
        
        for(int i=(mul*idx);i<mul*(idx+1);i=i+size+1){
            a[i] = al;    
        }
      }
      """)
      
    func = mod.get_function("doublify")
      
#    ran = 0.0000
#    start = time.time()      
      
    func(a_gpu, b_gpu, numpy.int32(len(word)), block=(thread_size,1,1),grid = (1, 1))
    
#    end = time.time()
#    ran += (end - start)
#    print "Total run time to replace = ", ran
      
    cuda.memcpy_dtoh(a, a_gpu)
    n = len(word)
    
    pycuda.tools.clear_context_caches()
    a_gpu.free()
    b_gpu.free()
    
    return [''.join(str(e) for e in a)[i:i+n] for i in range(0, len(a), n)]
