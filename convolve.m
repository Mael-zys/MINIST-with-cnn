%�������
function convolved = convolve(images,W,B) %W�Ǿ����Ȩ�أ�B��ƫ��

filter_dim=size(W,1);%�����ά��
filter1_num=size(W,3);%ǰһ�����˸���
filter2_num=size(W,4);%��ǰ����˸���

images_num = size(images,4);
image_dim = size(images,1);
conv_dim = image_dim-filter_dim+1;%���֮���ά��

%��ʼ��
convolved=zeros(conv_dim,conv_dim,filter2_num,images_num);

%���������Ĺ�ʽ��ԭ��д�ڱ����
for i=1:images_num %������ͼ��ѭ��
    for filtre2=1:filter2_num %�Ե�ǰ�ľ����ѭ��
        convolvedImage = zeros(conv_dim, conv_dim); %�ڶ�����������ά�ģ�ÿ�ξ��ȡ��i��ͼ��ĵ�j�������˵�j������Ȼ�����
        for filtre1=1:filter1_num %��i��ͼ��ĵ�j�������˵�j����
            filter = squeeze(W(:,:,filtre1,filtre2)); %��ȡ�������˵�Ȩ��
            filter = rot90(squeeze(filter),2);
            im = squeeze(images(:,:,filtre1,i));%��ȡ�������ͼ
            convolvedImage = convolvedImage + conv2(im,filter,'valid');%���
        end
        convolvedImage = bsxfun(@plus,convolvedImage,B(filtre2));%ƫ��
        convolvedImage = 1 ./ (1+exp(-convolvedImage));%sigmoid����
        convolved(:, :, filtre2, i) = convolvedImage;%��ɵ�i��ͼ���һ��filtre2�ľ��
    end
end
            
            
         