%卷积函数
function convolved = convolve(images,W,B) %W是卷积核权重，B是偏移

filter_dim=size(W,1);%卷积核维度
filter1_num=size(W,3);%前一层卷积核个数
filter2_num=size(W,4);%当前卷积核个数

images_num = size(images,4);
image_dim = size(images,1);
conv_dim = image_dim-filter_dim+1;%卷积之后的维度

%初始化
convolved=zeros(conv_dim,conv_dim,filter2_num,images_num);

%卷积（具体的公式和原理写在报告里）
for i=1:images_num %对所有图像循环
    for filtre2=1:filter2_num %对当前的卷积核循环
        convolvedImage = zeros(conv_dim, conv_dim); %第二层卷积核是三维的，每次卷积取第i个图像的第j面与卷积核第j面卷积，然后相加
        for filtre1=1:filter1_num %第i个图像的第j面与卷积核第j面卷积
            filter = squeeze(W(:,:,filtre1,filtre2)); %获取这个卷积核的权重
            filter = rot90(squeeze(filter),2);
            im = squeeze(images(:,:,filtre1,i));%获取待卷积的图
            convolvedImage = convolvedImage + conv2(im,filter,'valid');%卷积
        end
        convolvedImage = bsxfun(@plus,convolvedImage,B(filtre2));%偏移
        convolvedImage = 1 ./ (1+exp(-convolvedImage));%sigmoid函数
        convolved(:, :, filtre2, i) = convolvedImage;%完成第i个图像对一个filtre2的卷积
    end
end
            
            
         