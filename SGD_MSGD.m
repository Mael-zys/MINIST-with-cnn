%%当初是直接在SGD函数的基础上加minibatch改为MSGD的，忘记保留原来的SGD，不过要实验SGD也只需要把minibatch改为1就可以了
%初始设置
image_dim = 28; % 图像矩阵的大小
classes_num = 10;  % 十个数字十个类别
filter_dim1 = 5;    % 卷积核的维度
filter_dim2 = 5;
filters1_num = 10;  %卷积核数量
filters2_num = 10;
pool_dim1 = 2;      %池化维度 
pool_dim2 = 2;
alpha = 1.2*1e-1;       %学习率
epochs = 3; %训练次数
minibatch = 1;%设为1即SGD，实验MSGD，我们选的是150

conv1_dim=image_dim-filter_dim1+1;%第一层卷积过后的维度
pool1_dim=conv1_dim/pool_dim1;%第一层池化过后的维度
conv2_dim=pool1_dim-filter_dim2+1;%第二层卷积过后的维度
pool2_dim=conv2_dim/pool_dim2;%第二层池化过后的维度

%初始化卷积核
w1=1e-1*randn(filter_dim1,filter_dim1,1,filters1_num);
w2=1e-1*randn(filter_dim2,filter_dim2,filters1_num,filters2_num);

%初始化偏移
b1=zeros(filters1_num,1);
b2=zeros(filters2_num,1);

%softmax层
plan_size = pool2_dim^2*filters2_num;%把两次卷积后的结果展为平面的大小
%r  = sqrt(6) / sqrt(classes_num+plan_size+1);
%Wd = rand(numClasses, hiddenSize) * 2 * r - r;
ws=rand(classes_num,plan_size);%softmax层权重
bs=zeros(classes_num, 1);

%初始化改变量
delta_w1 = zeros(size(w1));
delta_b1 = zeros(size(b1));
delta_w2 = zeros(size(w2));
delta_b2 = zeros(size(b2));
delta_ws = zeros(size(ws));
delta_bs = zeros(size(bs));

it = 0;%迭代次数
C = [];%记录cost值的数组
A = [];

%读取训练集
images = read_image('train-images-idx3-ubyte');
images = reshape(images,image_dim,image_dim,1,[]);
labels = read_label('train-labels-idx1-ubyte');
labels(labels==0) = 10; %把0映射为10，不然之后sub2int会出错

l = length(labels); % 训练集的大小

for e=1:epochs
    
    order=randperm(l);%打乱顺序

    %把训练集分成minibatch
    for s=1:minibatch:(l-minibatch+1)
        it = it + 1;
        
        mb_images=images(:,:,:,order(s:s+minibatch-1));
        mb_labels=labels(order(s:s+minibatch-1));
        
        %在每轮训练一半的时候测试一下，多打几个点，方便看趋势
        if it == 30000 || it==90000 || it==150000
           test_images = read_image('t10k-images-idx3-ubyte');
           test_images = reshape(test_images,image_dim,image_dim,1,[]);
           test_labels = read_label('t10k-labels-idx1-ubyte');
           test_labels(test_labels==0) = 10;
           
           %两层卷积池化
           convolved1=convolve(test_images,w1,b1);
           pooled1=pool(convolved1,pool_dim1);
           convolved2=convolve(pooled1,w2,b2);
           pooled2=pool(convolved2,pool_dim2);
           
           %算概率
           pooled2=reshape(pooled2,[],length(test_images));
           probs = exp(bsxfun(@plus, ws * pooled2, bs));
           sum_pro=sum(probs,1);
           probs = bsxfun(@times, probs, 1 ./ sum_pro);
           [~,test_pro]=max(probs,[],1);
           test_pro=test_pro';
           
           %正确率
           taux=sum(test_pro==test_labels)/length(test_pro);
           fprintf('Accuracy is %f\n',taux);
           A(length(A)+1) = taux;
       end
        
        
        %初始化梯度
        w1_grad=zeros(size(w1));
        w2_grad=zeros(size(w2));
        ws_grad=zeros(size(ws));
        b1_grad=zeros(size(b1));
        b2_grad=zeros(size(b2));
        bs_grad=zeros(size(bs));
        
        %%   前馈传播(两层卷积池化）
        convolved1=convolve(mb_images,w1,b1);
        pooled1=pool(convolved1,pool_dim1);
        convolved2=convolve(pooled1,w2,b2);
        pooled2=pool(convolved2,pool_dim2);
        
        %%   Softmax
        pooled2=reshape(pooled2,[],minibatch);
        probs = exp(bsxfun(@plus, ws * pooled2, bs));
        sum_pro=sum(probs,1);
        probs = bsxfun(@rdivide, probs, sum_pro);
        
        %%   交叉熵算cost
        logp=log(probs);
        index = sub2ind(size(logp),mb_labels',1:size(probs,2));
        cost=-sum(logp(index))/minibatch;
     
       %%   BP算法
       %%  误差 (从后往前算）（具体公式原理在报告里）
        %softmax层误差
        output = zeros(size(probs));
        output(index) = 1;
        delta_softmax = probs -output;
        
        %第二层池化误差
        delta_pool2=reshape(ws'* delta_softmax,pool2_dim,pool2_dim,filters2_num,minibatch);
        delta_unpool2=zeros(conv2_dim,conv2_dim,filters2_num,minibatch);
        for nn=1:minibatch
            for ff=1:filters2_num
                unpool=delta_pool2(:,:,ff,nn);
                delta_unpool2(:,:,ff,nn)=kron(unpool,ones(pool_dim2))./(pool_dim2^2);
            end
        end
        
        %第二层卷积误差
        delta_conv2=delta_unpool2.*convolved2.*(1-convolved2);
        
        %第一层池化误差
        delta_pool1=zeros(pool1_dim,pool1_dim,filters1_num,minibatch);
        for i=1:minibatch
            for  f1=1:filters1_num
                for f2=1:filters2_num
                    delta_pool1(:,:,f1,i)=delta_pool1(:,:,f1,i)+convn(delta_conv2(:,:,f2,i),w2(:,:,f1,f2),'full');
                end
            end
        end
        
        delta_unpool1=zeros(conv1_dim,conv1_dim,filters1_num,minibatch);
        for nn=1:minibatch
            for ff=1:filters1_num
                unpool=delta_pool1(:,:,ff,nn);
                delta_unpool1(:,:,ff,nn)=kron(unpool,ones(pool_dim1))./(pool_dim1^2);
            end
        end
        
        %第一层卷积误差
        delta_conv1=delta_unpool1.*convolved1.*(1-convolved1);
       %%  梯度（具体公式原理在报告里）
        %ws,bs梯度
        ws_grad=delta_softmax*pooled2';
        bs_grad=sum(delta_softmax,2);
        
        %w2,b2梯度
        for f2=1:filters2_num
            for f1=1:filters1_num
                for i=1:minibatch
                    w2_grad(:,:,f1,f2)=w2_grad(:,:,f1,f2)+conv2(pooled1(:,:,f1,i),rot90(delta_conv2(:,:,f2,i),2),'valid');
                end
            end
            temp = delta_conv2(:,:,f2,:);
            b2_grad(f2) = sum(temp(:));
        end
        
        %w1,b1梯度
         for f1=1:filters1_num
                for i=1:minibatch
                    w1_grad(:,:,1,f1)=w1_grad(:,:,1,f1)+conv2(mb_images(:,:,1,i),rot90(delta_conv1(:,:,f1,i),2),'valid');
                end
            temp = delta_conv1(:,:,f1,:);
            b1_grad(f1) = sum(temp(:));
         end
        
         %计算改变量
        delta_ws=alpha*(ws_grad/minibatch);
        delta_bs=alpha*(bs_grad/minibatch);
        delta_w2=alpha*(w2_grad/minibatch);
        delta_b2=alpha*(b2_grad/minibatch);
        delta_w1=alpha*(w1_grad/minibatch);
        delta_b1=alpha*(b1_grad/minibatch);
        
        %更新权重和偏移
        ws=ws-delta_ws;
        bs=bs-delta_bs;
        w2=w2-delta_w2;
        b2=b2-delta_b2;
        w1=w1-delta_w1;
        b1=b1-delta_b1;
        
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
        C(length(C)+1) = cost;
    end

    %降低学习率
    alpha = alpha/2.0;
    
    %%     测试
    %读取数据
    test_images = read_image('t10k-images-idx3-ubyte');
    test_images = reshape(test_images,image_dim,image_dim,1,[]);
    test_labels = read_label('t10k-labels-idx1-ubyte');
    test_labels(test_labels==0) = 10;
    
    %两层卷积池化
    convolved1=convolve(test_images,w1,b1);
    pooled1=pool(convolved1,pool_dim1);
    convolved2=convolve(pooled1,w2,b2);
    pooled2=pool(convolved2,pool_dim2);
    
    %算概率
    pooled2=reshape(pooled2,[],length(test_images));
    probs = exp(bsxfun(@plus, ws * pooled2, bs));
    sum_pro=sum(probs,1);
    probs = bsxfun(@times, probs, 1 ./ sum_pro);
    [~,test_pro]=max(probs,[],1);
    test_pro=test_pro';
    
    %正确率
    taux=sum(test_pro==test_labels)/length(test_pro);
    fprintf('Accuracy is %f\n',taux);
    A(length(A)+1) = taux;
    plot(C);%cost的变化的图
    plot(A);%输出正确率的图
end

