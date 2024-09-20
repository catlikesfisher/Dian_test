import numpy as np
import matplotlib.pyplot as plt
from batch import select_batch
plt.rcParams['font.sans-serif'] = ['SimHei']
# RNN结构
#全程使用numpy,plt手搓,测试集准确率运气不好50%，好60%,建议多跑几遍呜呜呜
class rnn:
    def __init__(self, train_array,train_label,output_type_num):
        self.train_size = train_array.shape[0]  # 数据量，train60000,test10000
        self.input_shape = train_array.shape[1] * train_array.shape[2]
        self.train_array = train_array.reshape((self.train_size,self.input_shape,1))
        self.train_label = train_label
        self.output_type_num = output_type_num  #label类型，这里10类
        self.k = 28 #特征数的根号,这里也是隐藏层的行数
        self.U = np.random.normal(0, 2 / self.k, size=(self.k, self.input_shape))
        self.W = np.random.normal(0, 2 / self.k, size=(self.k, self.k))
        self.V = np.random.normal(0, 2 / self.k, size=(self.output_type_num, self.k))
        self.o = np.zeros((self.train_size, self.output_type_num))  # 初始化o
        self.s = np.zeros((self.train_size,self.k,1))  # 初始化s
        self.S = np.random.normal(0, 2 / self.k, size=(self.k, 1)) #初始化s_0
        self.s_pre = np.zeros((self.train_size,self.k,1))  # 初始化s_pre
        self.s_diag = np.zeros((self.train_size,self.k,self.k)) #使用s做一点变化，后面有用到几次
        self.b = np.random.normal(0, 2 / self.k, size=(self.k, 1))
        self.c = np.random.normal(0, 2 / self.k, size=(output_type_num, 1))#b,c为偏置参数
        self.grad_ot_L = np.zeros((self.train_size, self.output_type_num)) #初始化grad_ot_L
        self.grad_c_L = np.zeros((self.output_type_num,1)) #初始化grad_ot_L
        self.grad_s_L = np.zeros((self.train_size,self.k,1))
        self.grad_b_L = np.zeros((self.k,1))
        self.grad_V_L = np.zeros((self.output_type_num,self.k))
        self.grad_W_L = np.zeros((self.k,self.k))
        self.grad_U_L = np.zeros((self.k,self.input_shape))
        self.alpha = 0.0001
    def forward(self):
        i = 0
        S =  self.S#初始化s_pre
        while (i < self.train_size):
            # 先计算s_t和o
            a_t = self.W @ S + self.U@self.train_array[i]+ self.b
            s_t = np.tanh(a_t)
            self.s[i] = s_t
            self.o[i, :] = (self.V @ s_t + self.c).reshape(self.output_type_num,)
            exp = np.exp(self.o[i, :])
            self.o[i, :] = exp / exp.sum()
            # 更新s_t,i
            i = i + 1
            S = s_t
        return
    def loss(self):
        pi_yi = np.zeros((self.train_size, 1))
        i = 0
        while i < self.train_size:
            yi = self.train_label[i]
            pi_yi[i, 0] = self.o[i, yi]
            i = i + 1
        return -np.log(pi_yi).sum()
    def prepare_for_back(self):
        #计算grad_ot_L & 计算s_giag
        y = self.train_label
        yhat = self.o
        for t in range(self.train_size):
                yt = y[t]
                self.grad_ot_L[t, yt] = yhat[t, yt] - 1
                diag = np.eye(28)-np.diag(self.s[t].flatten())*np.diag(self.s[t].flatten())
                self.s_diag[t] = diag
        self.s_pre[1:] = self.s[:-1]
        self.s_pre[0] = self.S
    def prepare_grad_s_L(self):
        t = self.train_size-1
        self.grad_s_L[t] = 0.000000001*(self.V.transpose()@self.grad_ot_L[t]).reshape((28,1))#防止梯度消失，无奈之举
        t = t-1
        while t>-1:
            part_1 = np.array(self.V.transpose()@self.grad_ot_L[t]).reshape((28,1))
            part_2 = self.W.transpose()@self.s_diag[t+1]@self.grad_s_L[t+1]
            self.grad_s_L[t] = 0.0000000000001*(part_1+part_2)#防止梯度消失，无奈之举
            t = t-1
        return self.grad_s_L
    def grad(self):
        self.grad_c_L = np.reshape(self.grad_ot_L.sum(axis=0),(self.output_type_num,1))
        grad_b_L = np.zeros((self.train_size,self.k,1))
        for t in range(self.train_size):
            grad_b_L[t] = self.s_diag[t]@self.grad_s_L[t]
        self.grad_b_L = grad_b_L.sum(axis = 0)
        grad_V_L = np.zeros((self.train_size, self.output_type_num, self.k))
        for t in range(self.train_size):
            grad_V_L[t] =np.array(self.grad_ot_L[t]).reshape(self.output_type_num,1)@(self.s[t].transpose())
        self.grad_V_L = grad_V_L.sum(axis = 0)
        grad_W_L = np.zeros((self.train_size,self.k,self.k))
        for t in range(self.train_size):
            grad_W_L[t] = self.s_diag[t]@(self.grad_s_L[t]@self.s_pre[t].transpose())
        self.grad_W_L = grad_W_L.sum(axis = 0)
        grad_U_L = np.zeros((self.train_size,self.k,self.input_shape))
        for t in range(self.train_size):
            grad_U_L[t] = self.s_diag[t]@self.grad_s_L[t]@(self.train_array[t].transpose())
        self.grad_U_L = grad_U_L.sum(axis = 0)
    def back(self):
        self.W -= self.alpha*self.grad_W_L
        self.U -= self.alpha*self.grad_U_L
        self.V -= self.alpha*self.grad_V_L
        self.b -= self.alpha*self.grad_b_L
        self.c -= self.alpha*self.grad_c_L
    def predict(self):
        a = np.argmax(self.o, axis=1)#认为概率最大的就是正确分类
        accury = (a == self.train_label).sum()/a.shape[0]
        return accury
    def train(self,epoch):
        i = 0
        ac = []
        while i<epoch:
            self.alpha = self.alpha/(1+i//10)
            self.forward()
            self.prepare_for_back()
            self.prepare_grad_s_L()
            self.grad()
            self.back()
            ac.append(self.predict())
            #手动early stop
            if (i>2):
                if (ac[i]<ac[i-1])&(ac[i-1]<ac[i-2]):
                    print("early stop")
                    break
            i+=1
    def test(self,test_array,test_label):
        i = 0
        S = self.S  # 初始化s_pre
        test_array = test_array.reshape((test_array.shape[0],test_array.shape[1]*test_array.shape[2],1))
        test_size = test_array.shape[0]
        o = np.zeros((test_size, self.output_type_num))  # 初始化o
        s = np.zeros((test_size, self.k, 1))  # 初始化s
        while (i < test_size):
            # 先计算s_t和o
            a_t = self.W @ S + self.U@test_array[i] + self.b
            s_t = np.tanh(a_t)
            s[i] = s_t
            o[i, :] = (self.V @ s_t + self.c).flatten()
            exp = np.exp(o[i, :])
            o[i, :] = exp / exp.sum()
            # 更新s_t,i
            i = i + 1
            S = s_t

        a = np.argmax(o, axis=1)#认为概率最大的就是正确分类
        accury = (a == test_label).sum() / a.shape[0]
        print("测试集准确率",accury)
        return o
class rnn_bs:
    def __init__(self,train_array,train_label,output_type_num):
        batch_size = select_batch(train_array.shape[0])
        self.bs = batch_size.work()
        self.train_array = train_array
        self.train_label = train_label
        self.output_type_num = output_type_num
    def train(self):
        i = 0
        while i<self.bs.shape[0]:
            print("batch:%d/%d"%(i+1,self.bs.shape[0]))
            worker = rnn(self.train_array[self.bs[i]], self.train_label[self.bs[i]], self.output_type_num)
            if i > 0:
                worker.U = self.U
                worker.W = self.W
                worker.V = self.V
                worker.b = self.b
                worker.c = self.c
                worker.S = self.S
                worker.k =self.k
            worker.train(20)
            self.U = worker.U
            self.W = worker.W
            self.V = worker.V
            self.b = worker.b
            self.c = worker.c
            self.S = worker.S
            self.k = worker.k
            i = i + 1

    def test(self, test_array, test_label):
        i = 0
        S = self.S  # 初始化s_pre
        test_array = test_array.reshape((test_array.shape[0], test_array.shape[1] * test_array.shape[2], 1))
        test_size = test_array.shape[0]
        o = np.zeros((test_size, self.output_type_num))  # 初始化o
        s = np.zeros((test_size, self.k, 1))  # 初始化s
        while (i < test_size):
            # 先计算s_t和o
            a_t = self.W @ S + self.U @ test_array[i] + self.b
            s_t = np.tanh(a_t)
            s[i] = s_t
            o[i, :] = (self.V @ s_t + self.c).flatten()
            exp = np.exp(o[i, :])
            o[i, :] = exp / exp.sum()
            # 更新s_t,i
            i = i + 1
            S = s_t

        a = np.argmax(o, axis=1)  # 认为概率最大的就是正确分类
        accury = (a == test_label).sum() / a.shape[0]
        print("测试集准确率", accury)
        return o
