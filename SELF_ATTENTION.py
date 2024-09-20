import numpy as np
from PE import Pe #引入位置编码
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
class multi_head_attention:
    def __init__(self,a,head_num):
        self.a = a
        self.head_num = head_num
        self.w_k = np.random.normal(0, 1, size=(head_num,a.shape[0], a.shape[0]))
        self.w_q = np.random.normal(0, 1, size=(head_num,a.shape[0], a.shape[0]))
        self.w_v = np.random.normal(0, 1, size=(head_num,1, a.shape[0]))
        self.alpha = np.zeros((head_num,a.shape[1],a.shape[1]))
        self.w_o = np.random.normal(0, 1, size=(1,head_num))
        self.o = np.zeros((head_num,a.shape[1]))
    def rope(self):
        r = Pe(self.a)
        self.a += r.ROPE().reshape(self.a.shape)
    def forward(self):
        for i in range(self.head_num):
            q = self.w_q[i] @ self.a
            k = self.w_k[i] @ self.a
            k = np.clip(k,-30,30)
            v = self.w_v[i] @ self.a
            # softmax
            exp = np.exp(-k.transpose() @ q/(np.linalg.norm(k)**0.5))
            self.alpha[i] = exp / exp.sum(axis=0)
            self.o[i] = (v @ self.alpha[i])
        return (self.w_o @ self.o).flatten()
    def work(self):
        self.rope()
        self.forward()
        return self.w_o @ self.o
if __name__ == "__main__":
    #x是随机矩阵
    x = np.random.normal(0, 1, size=(20,30))
    sample = multi_head_attention(x,3)
    print(sample.work())













