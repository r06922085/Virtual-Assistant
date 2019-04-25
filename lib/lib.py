import numpy as np
import codecs

def read_file(file_name):
    f = codecs.open(file_name, 'r', 'utf-8').read().split('\n')
    f = np.asarray(f)
    lines = len(codecs.open(file_name, 'r' ,'utf-8').readlines())
    f = f.reshape(lines, int(len(f)/lines))
    return f
    
def write_file(file_name):

    return codecs.open(file_name, 'w', 'utf-8')
    