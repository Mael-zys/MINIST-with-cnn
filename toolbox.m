%用工具箱函数实现的CNN
%读取训练集
images = read_image('train-images-idx3-ubyte');
images = reshape(images,28,28,1,[]);
labels = read_label('train-labels-idx1-ubyte');
labels(labels==0) = 10; %把0映射为10，不然之后sub2int会出错

test_images = read_image('t10k-images-idx3-ubyte');
test_images = reshape(test_images,28,28,1,[]);
test_labels = read_label('t10k-labels-idx1-ubyte');
test_labels(test_labels==0) = 10;

l = length(labels); % 训练集的大小

%卷积层
layers = [
    imageInputLayer([28 28 1],"Name","imageinput")
    convolution2dLayer([5 5],10,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    tanhLayer("Name","tanh_1")
    averagePooling2dLayer([5 5],"Name","avgpool2d_1","Padding","same")
    convolution2dLayer([5 5],10,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    tanhLayer("Name","tanh_2")
    averagePooling2dLayer([5 5],"Name","avgpool2d_2","Padding","same")
    fullyConnectedLayer(10,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

%% 训练神经网络

%转化为trainNetwork要求输入的格式
label = categorical(labels(1:60000));
test_label = categorical(test_labels(1:10000));
test={test_images,test_label};
% for i=1:600
%     image(i,1)={squeeze(images(:,:,1))};
% end

% 设置训练参数
opts = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...%初始学习率
    'Shuffle','every-epoch', ...
    'MaxEpochs',3,...%最大训练轮数
    'ValidationData', test, ...
    'ValidationFrequency',150,...%测试频率
    'MiniBatchSize',200,... %minibatch大小
    'Verbose',false, ...
    'Plots','training-progress');

% 训练神经网络，保存网络
net = trainNetwork(images, label,layers ,opts);

%save 'CSNet.mat' net  %可以选择保存训练结果


