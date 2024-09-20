import numpy as np
import matplotlib.pyplot as plt
class Pe:
    def __init__(self,x):
        self.x = x
        self.seq_len = x.shape[1]
        #奇数加上一排0向量变成偶数
        if (self.seq_len%2 !=0):
            self.x = np.concatenate((self.x,np.zeros((self.x.shape[0],1))),axis =1)
            self.seq_len += 1
        self.d_model = x.shape[0] #转化成32维向量
    def APE(self,plot = 0):
        pos = np.arange(self.seq_len)[:, np.newaxis]
        div_i_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe = np.zeros((self.seq_len, self.d_model))
        pe[:, 0::2] = np.sin(pos * div_i_term)
        pe[:, 1::2] = np.cos(pos * div_i_term)
        #self.x += pe
        while plot == 1:
            # 展示绝对位置编码
            plt.figure(figsize=(12, 8))
            plt.imshow(pe)
            plt.colorbar()
            plt.title("Absolute Positional Encoding")
            plt.xlabel("dim of vector")
            plt.ylabel("positons")
            plt.show()
            break
        return pe.transpose()
    def ROPE(self,plot = 0):
        q = self.x
        m = np.array(np.arange(1,q.shape[1]+1)).reshape(1,q.shape[1])

        length = q.shape[0]
        d = (length + 1) // 2
        theta = np.array(1 / (10000 * (-length)) * np.arange(1, d + 1)).reshape((d, 1))@m
        q_p1 = np.array(q[d:,:])
        q_p2 = np.array(q[:d,:])
        q_hat = np.concatenate((q_p1, q_p2), axis=0)
        cos_raw = np.cos(theta)
        sin_raw = np.sin(theta)
        cos = np.concatenate((cos_raw, cos_raw), axis=0)
        sin = np.concatenate((sin_raw, sin_raw), axis=0)
        pe = np.multiply(q, cos) + np.multiply(q_hat, sin)
        while plot == 1:
            # 展示绝对位置编码
            plt.figure(figsize=(12, 8))
            plt.imshow(pe)
            plt.colorbar()
            plt.title("RO Positional Encoding")
            plt.xlabel("dim of vector")
            plt.ylabel("positons")
            plt.show()
            break
        return pe
if __name__ == "__main__":
    x = np.random.normal(0, 1, size=(2,5))
    a = Pe(x)
    print(a.APE(plot = 0))
    print(a.ROPE(plot = 0))

















