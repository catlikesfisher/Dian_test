import numpy as np


class select_batch:
    def __init__(self,length,bs = 256):
        self.length = length
        self.bs = bs
    def work(self):
        k = self.length//self.bs
        j = self.length%self.bs
        if j!=0:
            k = k+1
        output = np.zeros((k,self.bs))
        for i in range(k-1):
            output[i,:] = np.arange(self.bs*i,self.bs*i+self.bs)
        output[-1,:] = np.arange(self.length-self.bs,self.length)
        return output.astype(int)

