# Dian_test
###### It is the answer for the question a team entitled Dian.
# 您好，感谢阅读我的结果，如果您读后对我感兴趣，qq:1564240065，data中是训练数据，导入数据是记得修改路径，否则可能报错
###### 谢谢你查看我的提交结果，下面我使用简单地几句话对我的结果进行解释,我使用numpy手搓了rnn,PE,attention,DDPM的所有内容。
###### rnn我做了三个方面的工作。第一，基础的rnn在给定的数据集上准确率为45-65%，为了提高效率，我对基础的rnn(RNN.py)进行了两个方面的改进，一（也就是三个方面中的第二）是分batch(batch.py+RNN_batch)进行学习，我原以为这会缓解梯度爆炸，但是结果却收效甚微，准确率仍然为45-60%。二（也是三个方面中的第三方面）是集成学习(bagging_RNN.py),使用集成学习，准确率稳定在60%以上（我的最好结果是62%），特别的，最终的测试结果写在main.py。
###### PE和attention的工作在PE.py，SELF_ATTENTION中，测试结果写在这两个文件中。
###### DDPM的内容见于RNN_for_DDPM(由于这里使用的rnn和第一问比loss函数发生变化，进行了部分的改写)和DDPM(DDPM.py)，DDPM使用了上面手动实现rnn,PE的结果。
###### 我是用的IDE是pycharm,__pycache__是其自动生成的，防止报错也一并上传，data中是训练数据，导入数据是记得修改路径，否则可能报错
# 下面分文件下介绍一下我的工作
### main.py
###### 这个文件中导入了数据，把rnn的三种形式（上面一段有提到）的结果展示出来了。
### RNN.py
###### 这个文件中存放了最基础的rnn，这个rnn是numpy手搓的，这个rnn中没有batch参数，由于数据集比较小，这样结果反而很好。我使用的loss是概率的负的对数和，但是评价指标是预测成功的准确率（因为前者可以求导数，后者有更直接的现实意义），为了防止梯度爆炸，对W矩阵求导做了一点处理,在数据集上准确率为45-65%
###### 我想你可能对我的代码的具体内容感兴趣（特别是反向传播），下面是RNN中唯一的class——rnn的内容
###### __init__ ：用于初始化各种参数，包括学习率（self.alpha），各种梯度（grad_x_L表示L对x求导），W，V，U（与题目中的WVU含义相同），和偏置参数（a，b矩阵）等参数
###### forward：正向传播计算输出self.o
###### loss:计算loss,我使用的loss是概率的负的对数和
###### prepare_for_back:为反向传播做准备，计算一些中间变量
###### prepare_grad_s_L：为反向传播做准备，计算grad_st_L（L对s_t求导）这个比较复杂的中间变量
###### grad：计算L对W,V,U,a,b的梯度
###### back(这个函数名称不太恰当，后面也不想改了)：使用计算的梯度更新参数W，V，U，a，b
###### predict(这个函数名称也不太恰当，后面也不想改了)：用于根据输出的o（每个样本对应的o_t都是一个10*1的概率向量，对应10种类型的概率）判断预测的分类
###### train:这个函数有参数epoch,表示训练epoch次，这个函数支持 early stop，训练之后会画出每次训练后的准确率
###### test:这个函数用于测试集的预测，返回预测准确率
###### 测试代码在main中
### batch.py 
###### 这个文件中有class select_batch，用于给输入的数据集分batch
### RNN_batch.py
###### 在class rnn基础上写了class rnn_bs，可以支持分批次学习的RNN
###### 这个文件时最后写的，头比较晕，代码有点赘余，但是觉得修改也没有必要就没做简化
###### 注意这个文件 from batch import select_batch了（select_batch是我自己写的一个简单的class，见batch.py）
###### 测试代码在main中
### bagging_RNN.py
###### 这个文件写的比较早，结构比较简明清楚，用于集成学习（用一群RNN学习）提高RNN准确率
###### 这个文件from RNN import rnn，引用了自制的rnn
###### 测试代码在main中


