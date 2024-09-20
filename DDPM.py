import torchvision  # 用于导入数据
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from RNN_FOR_DDPM import ddpm_rnn_bs
from PE import Pe
from SELF_ATTENTION import multi_head_attention
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


class ddpm:
    def __init__(self,train_array):
        #整理数据
        self.train_size = train_array.shape[0]  # 数据量，train6000,test1000
        self.train_array = train_array.reshape((self.train_size, train_array.shape[2], train_array.shape[3]))
        self.num_steps = 100
        self.beta = np.linspace(-5,5,self.num_steps)
        self.beta = 1/(1+np.exp(-self.beta))
        self.alpha = 1-self.beta
        self.alpha_prod = np.cumprod(self.alpha)
        self.alpha_prod_pre = np.concatenate((np.array([1]),self.alpha_prod),axis =0)
        self.alpha_bar_sqrt = np.sqrt(self.alpha_prod)
        self.one_minus_alpha_bar_log = np.log(1-self.alpha_prod)
        self.one_minus_alpha_bar_sqrt = np.sqrt(1-self.alpha_prod)
        #self.alpha_pre
        pass
    def q_x(self,t):
        noise = np.random.normal(0, 1, size=self.train_array.shape)
        alpha_t = self.alpha_bar_sqrt[t]
        alpha_l_m_t = self.one_minus_alpha_bar_sqrt[t]
        return alpha_t*self.train_array+alpha_l_m_t*noise
    def show_q_x(self,s=0):
        num_show = 10

        for i in range(num_show):
            plt.subplot(1, num_show, i+1)
            q_i = self.q_x(i*self.num_steps//num_show)[s]
            plt.imshow(q_i)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        #我的电脑这样最好看
        plt.title("q_x以一个样本为例子可视化",x = -5)
        plt.show()



    def loss(self):
        t = np.random.randint(10,self.num_steps)
        #计算根号下alpha_bar
        a = self.alpha_bar_sqrt[t]
        # 计算根号下1-alpha_bar
        am1 = self.one_minus_alpha_bar_sqrt[t]
        error = np.random.normal(0, 1, size=self.train_array.shape)
        x = self.train_array*a+error*am1
        #训练参数

        t = np.random.randint(1,self.num_steps)
        self.shape_another = self.train_array.shape #备用
        train_x = np.zeros((self.train_array.shape[0],self.train_array.shape[1],self.train_array.shape[2]+1))
        train_e = np.zeros((self.train_array.shape[0],self.train_array.shape[1],self.train_array.shape[2]+1))
        train_xx = np.zeros((self.train_array.shape[0], self.train_array.shape[1], self.train_array.shape[2] + 2))
        train_ee = np.zeros((self.train_array.shape[0], self.train_array.shape[1], self.train_array.shape[2] + 2))
        for i in range(self.train_array.shape[0]):
            train_x[i] = np.concatenate((x[i],t*np.ones((self.train_array.shape[1],1))),axis=1)
            train_e[i] = np.concatenate((error[i],np.zeros((self.train_array.shape[1],1))),axis=1)
            train_xx[i] = np.concatenate((train_x[i], np.zeros((self.train_array.shape[1], 1))), axis=1)#保证为偶数
            train_ee[i] = np.concatenate((train_e[i], np.zeros((self.train_array.shape[1], 1))), axis=1)#保证为偶数
        #train_xx = train_xx.reshape((train_xx.shape[0],train_xx.shape[1],1))
        train_array = train_xx
        for i in range(train_array.shape[0]):
            r = Pe(train_array[i])
            train_array[i] += r.ROPE()
        self.train_array_shape_last = train_array.shape
        train_array = train_array.reshape((train_array.shape[0], train_array.shape[2] * train_array.shape[1]))
        e = train_ee.reshape((train_ee.shape[0], train_ee.shape[2] * train_ee.shape[1]))
        model = ddpm_rnn_bs(train_array,e,bs_raw = train_array.shape[0])
        self.e_shape = e.shape
        model.train()
        #reverse
        x_T = np.random.normal(0,1,size=self.e_shape)
        for t in range(self.num_steps,0):
            if t>1:
                z = np.random.normal(0,1,size=self.e_shape)
            else:
                z = np.zeros(self.e_shape)
            x_T = (self.alpha**0.5)*(x_T-(1-self.alpha)/((1-self.alpha_prod)**0.5)*model.forcast())+z*self.beta


        y_T = x_T.reshape(self.train_array_shape_last)
        j_T = np.zeros(self.shape_another)
        for i in range(x_T.shape[0]):
            j_T[i] = (y_T[i][:,:-2]).reshape(self.shape_another[1],self.shape_another[2])
        return j_T














#选出一部分
index =  np.argwhere(train_label == 1)
train_array = train_array[index]
train_label = train_label[index]
#test同理
index_test = np.argwhere(test_label == 1)
test_array = test_array[index_test]
test_label = test_label[index_test]

a = ddpm(train_array)
a.q_x(60)
#a.show_q_x()
result = a.loss()
plt.imshow(result[0])
plt.show()


