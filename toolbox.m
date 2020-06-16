%�ù����亯��ʵ�ֵ�CNN
%��ȡѵ����
images = read_image('train-images-idx3-ubyte');
images = reshape(images,28,28,1,[]);
labels = read_label('train-labels-idx1-ubyte');
labels(labels==0) = 10; %��0ӳ��Ϊ10����Ȼ֮��sub2int�����

test_images = read_image('t10k-images-idx3-ubyte');
test_images = reshape(test_images,28,28,1,[]);
test_labels = read_label('t10k-labels-idx1-ubyte');
test_labels(test_labels==0) = 10;

l = length(labels); % ѵ�����Ĵ�С

%�����
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

%% ѵ��������

%ת��ΪtrainNetworkҪ������ĸ�ʽ
label = categorical(labels(1:60000));
test_label = categorical(test_labels(1:10000));
test={test_images,test_label};
% for i=1:600
%     image(i,1)={squeeze(images(:,:,1))};
% end

% ����ѵ������
opts = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...%��ʼѧϰ��
    'Shuffle','every-epoch', ...
    'MaxEpochs',3,...%���ѵ������
    'ValidationData', test, ...
    'ValidationFrequency',150,...%����Ƶ��
    'MiniBatchSize',200,... %minibatch��С
    'Verbose',false, ...
    'Plots','training-progress');

% ѵ�������磬��������
net = trainNetwork(images, label,layers ,opts);

%save 'CSNet.mat' net  %����ѡ�񱣴�ѵ�����


