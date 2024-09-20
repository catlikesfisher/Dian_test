#from torch import tensor #本来打算使用的，后面发现没有用上，用numpy就够了
import torchvision  # 用于导入数据
from torchvision import transforms
from RNN import rnn
from batch import select_batch
from bagging_RNN import bagging_rnn#集成学习效果更好
from RNN_batch import rnn_bs
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()
# 读取数据
train = torchvision.datasets.FashionMNIST(root=r"..\Dian\data", train=True, transform=trans, download=False)
test = torchvision.datasets.FashionMNIST(root=r"..\Dian\data", train=False, transform=trans, download=False)
input_size = train.data[0].numel()  # input_size = 行数*列数=28*28=784
output_type_num = 10  # 10个种类

#训练
train_array = train.data.numpy()/255
train_label = train.targets.numpy()
test_array = test.data.numpy()/255
test_label = test.targets.numpy()

#普通的rnn,效果45-60%
#a = rnn(train_array,train_label,10)
#a.train(10)

#a.test(test_array,test_label)

#集成学习，效果稳定在60%以上
#b = bagging_rnn(train_array,train_label,test_array,test_label,10,size = 2,epoch = 10)
#b.run()


#分batch的rnn,效果45-60%
batch_size = rnn_bs(train_array,train_label,10)
print(batch_size.train())
batch_size.test(test_array,test_label)


#选出一部分
index =  np.argwhere(train_label == 1)
train_array = train_array[index]
train_label = train_label[index]
#test同理
index_test = np.argwhere(test_label == 1)
test_array = test_array[index_test]
test_label = test_label[index_test]

#DDPM
a = ddpm(train_array)
a.q_x(60)
#a.show_q_x()
print(a.loss())
