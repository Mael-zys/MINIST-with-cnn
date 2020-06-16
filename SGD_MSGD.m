%%������ֱ����SGD�����Ļ����ϼ�minibatch��ΪMSGD�ģ����Ǳ���ԭ����SGD������Ҫʵ��SGDҲֻ��Ҫ��minibatch��Ϊ1�Ϳ�����
%��ʼ����
image_dim = 28; % ͼ�����Ĵ�С
classes_num = 10;  % ʮ������ʮ�����
filter_dim1 = 5;    % ����˵�ά��
filter_dim2 = 5;
filters1_num = 10;  %���������
filters2_num = 10;
pool_dim1 = 2;      %�ػ�ά�� 
pool_dim2 = 2;
alpha = 1.2*1e-1;       %ѧϰ��
epochs = 3; %ѵ������
minibatch = 1;%��Ϊ1��SGD��ʵ��MSGD������ѡ����150

conv1_dim=image_dim-filter_dim1+1;%��һ���������ά��
pool1_dim=conv1_dim/pool_dim1;%��һ��ػ������ά��
conv2_dim=pool1_dim-filter_dim2+1;%�ڶ����������ά��
pool2_dim=conv2_dim/pool_dim2;%�ڶ���ػ������ά��

%��ʼ�������
w1=1e-1*randn(filter_dim1,filter_dim1,1,filters1_num);
w2=1e-1*randn(filter_dim2,filter_dim2,filters1_num,filters2_num);

%��ʼ��ƫ��
b1=zeros(filters1_num,1);
b2=zeros(filters2_num,1);

%softmax��
plan_size = pool2_dim^2*filters2_num;%�����ξ����Ľ��չΪƽ��Ĵ�С
%r  = sqrt(6) / sqrt(classes_num+plan_size+1);
%Wd = rand(numClasses, hiddenSize) * 2 * r - r;
ws=rand(classes_num,plan_size);%softmax��Ȩ��
bs=zeros(classes_num, 1);

%��ʼ���ı���
delta_w1 = zeros(size(w1));
delta_b1 = zeros(size(b1));
delta_w2 = zeros(size(w2));
delta_b2 = zeros(size(b2));
delta_ws = zeros(size(ws));
delta_bs = zeros(size(bs));

it = 0;%��������
C = [];%��¼costֵ������
A = [];

%��ȡѵ����
images = read_image('train-images-idx3-ubyte');
images = reshape(images,image_dim,image_dim,1,[]);
labels = read_label('train-labels-idx1-ubyte');
labels(labels==0) = 10; %��0ӳ��Ϊ10����Ȼ֮��sub2int�����

l = length(labels); % ѵ�����Ĵ�С

for e=1:epochs
    
    order=randperm(l);%����˳��

    %��ѵ�����ֳ�minibatch
    for s=1:minibatch:(l-minibatch+1)
        it = it + 1;
        
        mb_images=images(:,:,:,order(s:s+minibatch-1));
        mb_labels=labels(order(s:s+minibatch-1));
        
        %��ÿ��ѵ��һ���ʱ�����һ�£���򼸸��㣬���㿴����
        if it == 30000 || it==90000 || it==150000
           test_images = read_image('t10k-images-idx3-ubyte');
           test_images = reshape(test_images,image_dim,image_dim,1,[]);
           test_labels = read_label('t10k-labels-idx1-ubyte');
           test_labels(test_labels==0) = 10;
           
           %�������ػ�
           convolved1=convolve(test_images,w1,b1);
           pooled1=pool(convolved1,pool_dim1);
           convolved2=convolve(pooled1,w2,b2);
           pooled2=pool(convolved2,pool_dim2);
           
           %�����
           pooled2=reshape(pooled2,[],length(test_images));
           probs = exp(bsxfun(@plus, ws * pooled2, bs));
           sum_pro=sum(probs,1);
           probs = bsxfun(@times, probs, 1 ./ sum_pro);
           [~,test_pro]=max(probs,[],1);
           test_pro=test_pro';
           
           %��ȷ��
           taux=sum(test_pro==test_labels)/length(test_pro);
           fprintf('Accuracy is %f\n',taux);
           A(length(A)+1) = taux;
       end
        
        
        %��ʼ���ݶ�
        w1_grad=zeros(size(w1));
        w2_grad=zeros(size(w2));
        ws_grad=zeros(size(ws));
        b1_grad=zeros(size(b1));
        b2_grad=zeros(size(b2));
        bs_grad=zeros(size(bs));
        
        %%   ǰ������(�������ػ���
        convolved1=convolve(mb_images,w1,b1);
        pooled1=pool(convolved1,pool_dim1);
        convolved2=convolve(pooled1,w2,b2);
        pooled2=pool(convolved2,pool_dim2);
        
        %%   Softmax
        pooled2=reshape(pooled2,[],minibatch);
        probs = exp(bsxfun(@plus, ws * pooled2, bs));
        sum_pro=sum(probs,1);
        probs = bsxfun(@rdivide, probs, sum_pro);
        
        %%   ��������cost
        logp=log(probs);
        index = sub2ind(size(logp),mb_labels',1:size(probs,2));
        cost=-sum(logp(index))/minibatch;
     
       %%   BP�㷨
       %%  ��� (�Ӻ���ǰ�㣩�����幫ʽԭ���ڱ����
        %softmax�����
        output = zeros(size(probs));
        output(index) = 1;
        delta_softmax = probs -output;
        
        %�ڶ���ػ����
        delta_pool2=reshape(ws'* delta_softmax,pool2_dim,pool2_dim,filters2_num,minibatch);
        delta_unpool2=zeros(conv2_dim,conv2_dim,filters2_num,minibatch);
        for nn=1:minibatch
            for ff=1:filters2_num
                unpool=delta_pool2(:,:,ff,nn);
                delta_unpool2(:,:,ff,nn)=kron(unpool,ones(pool_dim2))./(pool_dim2^2);
            end
        end
        
        %�ڶ��������
        delta_conv2=delta_unpool2.*convolved2.*(1-convolved2);
        
        %��һ��ػ����
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
        
        %��һ�������
        delta_conv1=delta_unpool1.*convolved1.*(1-convolved1);
       %%  �ݶȣ����幫ʽԭ���ڱ����
        %ws,bs�ݶ�
        ws_grad=delta_softmax*pooled2';
        bs_grad=sum(delta_softmax,2);
        
        %w2,b2�ݶ�
        for f2=1:filters2_num
            for f1=1:filters1_num
                for i=1:minibatch
                    w2_grad(:,:,f1,f2)=w2_grad(:,:,f1,f2)+conv2(pooled1(:,:,f1,i),rot90(delta_conv2(:,:,f2,i),2),'valid');
                end
            end
            temp = delta_conv2(:,:,f2,:);
            b2_grad(f2) = sum(temp(:));
        end
        
        %w1,b1�ݶ�
         for f1=1:filters1_num
                for i=1:minibatch
                    w1_grad(:,:,1,f1)=w1_grad(:,:,1,f1)+conv2(mb_images(:,:,1,i),rot90(delta_conv1(:,:,f1,i),2),'valid');
                end
            temp = delta_conv1(:,:,f1,:);
            b1_grad(f1) = sum(temp(:));
         end
        
         %����ı���
        delta_ws=alpha*(ws_grad/minibatch);
        delta_bs=alpha*(bs_grad/minibatch);
        delta_w2=alpha*(w2_grad/minibatch);
        delta_b2=alpha*(b2_grad/minibatch);
        delta_w1=alpha*(w1_grad/minibatch);
        delta_b1=alpha*(b1_grad/minibatch);
        
        %����Ȩ�غ�ƫ��
        ws=ws-delta_ws;
        bs=bs-delta_bs;
        w2=w2-delta_w2;
        b2=b2-delta_b2;
        w1=w1-delta_w1;
        b1=b1-delta_b1;
        
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
        C(length(C)+1) = cost;
    end

    %����ѧϰ��
    alpha = alpha/2.0;
    
    %%     ����
    %��ȡ����
    test_images = read_image('t10k-images-idx3-ubyte');
    test_images = reshape(test_images,image_dim,image_dim,1,[]);
    test_labels = read_label('t10k-labels-idx1-ubyte');
    test_labels(test_labels==0) = 10;
    
    %�������ػ�
    convolved1=convolve(test_images,w1,b1);
    pooled1=pool(convolved1,pool_dim1);
    convolved2=convolve(pooled1,w2,b2);
    pooled2=pool(convolved2,pool_dim2);
    
    %�����
    pooled2=reshape(pooled2,[],length(test_images));
    probs = exp(bsxfun(@plus, ws * pooled2, bs));
    sum_pro=sum(probs,1);
    probs = bsxfun(@times, probs, 1 ./ sum_pro);
    [~,test_pro]=max(probs,[],1);
    test_pro=test_pro';
    
    %��ȷ��
    taux=sum(test_pro==test_labels)/length(test_pro);
    fprintf('Accuracy is %f\n',taux);
    A(length(A)+1) = taux;
    plot(C);%cost�ı仯��ͼ
    plot(A);%�����ȷ�ʵ�ͼ
end

