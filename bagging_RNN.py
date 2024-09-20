from RNN import rnn
import numpy as np
import pandas as pd
class bagging_rnn:
    def __init__(self,train_array,train_label,test_array,test_label,output_type_num,size,epoch):
        self.size = size
        self.epoch = epoch
        self.train_array = train_array
        self.train_label = train_label
        self.test_array = test_array
        self.test_label = test_label
        self.out_put_type = output_type_num
        self.ans = np.zeros((test_array.shape[0],self.size))
        self.final_ans = np.zeros((test_array.shape[0],1))
    def learn(self):
        for i in range(self.size):
            print("集成学习进行中，进度为%d/%d,下面为第%d次学习的轮次和结果"%(i+1,self.size,i+1))
            a = rnn(self.train_array, self.train_label, self.out_put_type)
            a.train(self.epoch)
            self.ans[:,i] = np.argmax(a.test(self.test_array,self.test_label), axis=1)
    def summary(self):
        data = pd.DataFrame(self.ans)
        for i in range(self.test_array.shape[0]):
            # 返回众数
            self.final_ans[i] = data.iloc[i,:].mode()[0]
    def accuray(self):
        ac = (self.final_ans.transpose() == self.test_label).sum()/ self.test_label.shape[0]
        print("集成学习测试集准确度",ac)
        return ac
    def run(self):
        self.learn()
        self.summary()
        self.accuray()




