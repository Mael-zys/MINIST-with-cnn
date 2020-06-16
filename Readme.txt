我们用的双层卷积神经网络，后向传播用的是随机梯度下降及其优化版本

函数说明：
read_label和read_image分别为读取标签和图像数据点的函数
convolve是实现卷积的函数，pool是实现池化的函数
SGD_MSGD是主函数，可以直接运行得到答案(把minibatch设为1就是SGD，大于1就是MSGD）
OPTIMAL是优化版的主函数，可以直接运行得到答案
OPTIMAL_FINALE是最终优化版的主函数，可以直接运行得到答案
toolbox是用工具箱函数写的CNN，可以直接运行得到答案

运行效果对比：(toolbox函数一直没变，只是改了minibatch来和自己的算法对比）
（在文件夹“实验图”和我们的报告里都有相应的图）

1.
SGD：（最基本的随机梯度下降，每输入一个图像就更新一次）
经过三轮训练，准确率97.99%(耗时较长)

toolbox:
minibatch=1，经过三轮训练，准确率94.05%（不建议尝试，这个耗时90分钟）

2.
MSGD：（增加了minibatch，改为对minibatch随机梯度下降）
minibatch=150，经过三轮训练，准确率93.74%

toolbox:（minibatch=150)
minibatch=150，经过三轮训练，准确率98.36%（调整初始学习率之后）

3.
优化版本OPTIMAL：（在修改版基础上加了动量和权重衰减）
minibatch=150，经过三轮训练，准确率97.91%

toolbox:
minibatch=150，经过三轮训练，准确率98.36%（调整初始学习率之后）

4.
（1）
最终优化版本OPTIMAL_FINALE：（在修改版基础上加了Adam算法，可以自动调整学习率）
minibatch=150，经过三轮训练，准确率98.02%

toolbox:
minibatch=150，经过三轮训练，准确率98.36%（调整初始学习率之后）

（2）
最终优化版本OPTIMAL_FINALE：（在修改版基础上加了Adam算法，可以自动调整学习率）
minibatch=200，经过三轮训练，准确率98.38%

toolbox:
minibatch=200，经过三轮训练，准确率98.32%（调整初始学习率之后）