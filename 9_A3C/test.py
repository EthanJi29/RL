import numpy as np
import multiprocessing
a = [[1,2,3]]
a = np.array(a)
a = a[np.newaxis,:]
print(a)
print(multiprocessing.cpu_count())