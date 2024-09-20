import numpy as np
import matplotlib.pyplot as plt
from SELF_ATTENTION import multi_head_attention
from batch import select_batch
plt.rcParams['font.sans-serif'] = ['SimHei']
# RNN结构
#因为之前rnn准门设计解决分类问题，要输入label,现在对他进行改写，重点改写如下
#删除了关于label的部分
#改变了loss函数
class ddpm_rnn:
    def __init__(self, train_array, ep):
        #输入矩阵，以及误差ep
        self.train_size = train_array.shape[0]  # 数据量，train60000,test10000
        self.input_shape = train_array.shape[1]
        self.train_array = train_array.reshape((self.train_size, self.input_shape, 1))
        self.ep = ep #ep输入要求：batch*output_num
        self.k = 28 #特征数的根号,这里也是隐藏层的行数
        self.output_num = self.ep.shape[1] #之前的output_type_num没用了，换成ep的维度
        #下面的都可以接着用
        self.U = np.random.normal(0, 2 / self.k, size=(self.k, self.input_shape))
        self.W = np.random.normal(0, 2 / self.k, size=(self.k, self.k))
        self.V = np.random.normal(0, 2 / self.k, size=(self.output_num, self.k))
        self.o = np.zeros((self.train_size, self.output_num))  # 初始化o
        self.s = np.zeros((self.train_size,self.k,1))  # 初始化s
        self.S = np.random.normal(0, 2 / self.k, size=(self.k, 1)) #初始化s_0
        self.s_pre = np.zeros((self.train_size,self.k,1))  # 初始化s_pre
        self.s_diag = np.zeros((self.train_size,self.k,self.k)) #使用s做一点变化，后面有用到几次
        self.b = np.random.normal(0, 2 / self.k, size=(self.k, 1))
        self.c = np.random.normal(0, 2 / self.k, size=(self.output_num, 1))#b,c为偏置参数
        self.grad_ot_L = np.zeros((self.train_size, self.output_num)) #初始化grad_ot_L
        self.grad_c_L = np.zeros((self.output_num, 1)) #初始化grad_ot_L
        self.grad_s_L = np.zeros((self.train_size,self.k,1))
        self.grad_b_L = np.zeros((self.k,1))
        self.grad_V_L = np.zeros((self.output_num, self.k))
        self.grad_W_L = np.zeros((self.k,self.k))
        self.grad_U_L = np.zeros((self.k,self.input_shape))
        self.alpha = 0.0001
        self.l = 0
    def forward(self):
        i = 0
        S =  self.S#初始化s_pre
        while (i < self.train_size):
            # 先计算s_t和o
            a_t = self.W @ S + self.U@self.train_array[i]+ self.b
            s_t = np.tanh(a_t)
            self.s[i] = s_t
            self.o[i, :] = (self.V @ s_t + self.c).reshape(self.output_num, )
            exp = np.exp(self.o[i, :])
            self.o[i, :] = exp / exp.sum()
            # 更新s_t,i
            i = i + 1
            S = s_t
        return
    def loss(self):
        self.l = ((self.ep-self.o)*(self.ep-self.o)).sum()
    def prepare_for_back(self):
        #计算grad_ot_L & 计算s_giag
        self.grad_ot_L = 2*(self.o-self.ep)
        for t in range(self.train_size):
                diag = np.eye(28)-np.diag(self.s[t].flatten())*np.diag(self.s[t].flatten())
                self.s_diag[t] = diag
        self.s_pre[1:] = self.s[:-1]
        self.s_pre[0] = self.S
    def prepare_grad_s_L(self):
        t = self.train_size-1
        self.grad_s_L[t] = 0.0000000001*(self.V.transpose()@self.grad_ot_L[t]).reshape((28,1))#防止梯度消失，无奈之举
        t = t-1
        while t>-1:
            part_1 = np.array(self.V.transpose()@self.grad_ot_L[t]).reshape((28,1))
            part_2 = self.W.transpose()@self.s_diag[t+1]@self.grad_s_L[t+1]
            self.grad_s_L[t] = 0.00000000000001*(part_1+part_2)#防止梯度消失，无奈之举
            t = t-1
        return self.grad_s_L
    def grad(self):
        self.grad_c_L = np.reshape(self.grad_ot_L.sum(axis=0), (self.output_num, 1))
        grad_b_L = np.zeros((self.train_size,self.k,1))
        for t in range(self.train_size):
            grad_b_L[t] = self.s_diag[t]@self.grad_s_L[t]
        self.grad_b_L = grad_b_L.sum(axis = 0)
        grad_V_L = np.zeros((self.train_size, self.output_num, self.k))
        for t in range(self.train_size):
            grad_V_L[t] = np.array(self.grad_ot_L[t]).reshape(self.output_num, 1) @ (self.s[t].transpose())
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

    def train(self,epoch):
        i = 0
        ac = []
        while i<epoch:
            self.alpha = self.alpha/(1+i//10)
            self.forward()
            self.prepare_for_back()
            #self.prepare_grad_s_L()
            self.grad()
            self.back()
            self.loss()
            ac.append(self.l)

            print("epoch:%d/%d===>训练集loss:%f"%(i+1,epoch,ac[i]))
            #手动early stop
            #if (i>2):
            #    if (ac[i]<ac[i-1])&(ac[i-1]<ac[i-2]):
            #        print("early stop")
            #        break
            i+=1
        #plt.plot(range(min(i+1,epoch)),ac)
        #plt.title("loss随时间变化图")
        #plt.show()
        self.tmp_loss = ac[-1]
class ddpm_rnn_bs:
    def __init__(self,train_array,ep,bs_raw = 64):
        batch_size = select_batch(bs =bs_raw ,length=train_array.shape[0]) #这由于输入数据的原因，是shape[1]
        self.bs = batch_size.work()
        self.train_array = train_array
        self.ep = ep
        self.output_num = self.ep.shape[1]  # 之前的output_type_num没用了，换成ep的维度
    def train(self):
        i = 0
        li = []
        while i<self.bs.shape[0]:
            print("batch:%d/%d"%(i+1,self.bs.shape[0]))
            worker = ddpm_rnn(self.train_array[self.bs[i]], self.ep[self.bs[i]])
            if i > 0:
                worker.U = self.U
                worker.W = self.W
                worker.V = self.V
                worker.b = self.b
                worker.c = self.c
                worker.S = self.S
                worker.k =self.k
            worker.train(100)
            self.U = worker.U
            self.W = worker.W
            self.V = worker.V
            self.b = worker.b
            self.c = worker.c
            self.S = worker.S
            self.k = worker.k
            self.o = worker.o
            li.append(worker.tmp_loss)
            i = i + 1
        if i>2:
            plt.plot(range(self.bs.shape[0]),np.array(li))
            plt.title("loss图")
            plt.xlabel("训练次数")
            plt.ylabel("loss")
            plt.legend(labels=['loss'])
            plt.show()
    def forcast(self,train_size):
        i = 0
        S = self.S  # 初始化s_pre
        while (i < train_size):
                # 先计算s_t和o
            a_t = self.W @ S + self.U @ self.train_array[i] + self.b
            s_t = np.tanh(a_t)
            self.s[i] = s_t
            self.o[i, :] = (self.V @ s_t + self.c).reshape(self.output_num, )
            exp = np.exp(self.o[i, :])
            self.o[i, :] = exp / exp.sum()
            # 更新s_t,i
            i = i + 1
            S = s_t
        return o
if __name__ =="__main__":

    import torchvision  # 用于导入数据
    from torchvision import transforms
    plt.rcParams['font.sans-serif'] = ['SimHei']
    trans = transforms.ToTensor()
    # 读取数据
    train = torchvision.datasets.FashionMNIST(root=r"..\Dian\data", train=True, transform=trans, download=False)
    test = torchvision.datasets.FashionMNIST(root=r"..\Dian\data", train=False, transform=trans, download=False)
    input_size = train.data[0].numel()  # input_size = 行数*列数=28*28=784
    output_type_num = 10  # 10个种类
    train_array = train.data.numpy()/255
    train_label = train.targets.numpy()
    #a = rnn(train_array,train_label,10)
    #a.train(10)
    test_array = test.data.numpy()/255
    test_label = test.targets.numpy()
    #选出一部分
    index =  np.argwhere(train_label == 1)
    train_array = train_array[index]
    train_label = train_label[index]
    #test同理
    index_test = np.argwhere(test_label == 1)
    test_array = test_array[index_test]
    test_label = test_label[index_test]


    train_array = train_array.reshape((train_array.shape[0], train_array.shape[2],train_array.shape[3]))
    train_array_mod = np.zeros(train_array.shape)
    for i in range(train_array.shape[0]):
        sample = multi_head_attention(train_array[i],3)
        train_array[i] = sample.work().transpose()
    train_array = train_array.reshape((train_array.shape[0], train_array.shape[2]*train_array.shape[1]))
    ep = np.random.normal(0, 1, train_array.shape)
    model = ddpm_rnn_bs(train_array,ep)
    model.train()

