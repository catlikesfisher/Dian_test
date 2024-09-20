# Dian_test
###### It is the answer for the question a team entitled Dian.
# 您好，感谢阅读我的结果，如果您读后对我感兴趣，qq:1564240065
谢谢你查看我的提交结果，下面我使用简单地几句话对我的结果进行解释,我使用numpy手搓了rnn,PE,attention,DDPM的所有内容。
###### rnn我做了三个方面的工作。第一，基础的rnn在给定的数据集上准确率为45-65%，为了提高效率，我对基础的rnn(RNN.py)进行了两个方面的改进，一（也就是三个方面中的第二）是分batch(batch.py+RNN_batch)进行学习，我原以为这会缓解梯度爆炸，但是结果却收效甚微，准确率仍然为45-60%。二（也是三个方面中的第三方面）是集成学习(bagging_RNN.py),使用集成学习，准确率稳定在60%以上（我的最好结果是62%），特别的，最终的测试结果写在main.py。
###### PE和attention的工作在PE.py,SELF_ATTENTION中，测试结果写在这两个文件中。
###### DDPM的内容见于RNN_for_DDPM(由于这里使用的rnn和第一问比loss函数发生变化，进行了部分的改写)和DDPM(DDPM.py)，DDPM使用了上面手动实现rnn,PE的结果。
# 下面分文件下介绍一下我的工作
