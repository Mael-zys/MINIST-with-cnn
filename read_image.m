function images = read_image(filename)
%��ȡ���ص㣬ת��Ϊ28x28x60000�ľ����28x28x10000�ľ���
fp = fopen(filename,'rb');
magic = fread(fp,1,'int32',0,'ieee-be');%�ļ���һλ��ħ��
num_images = fread(fp,1,'int32',0,'ieee-be');%�ڶ�λ��ͼ�����
num_rows = fread(fp,1,'int32',0,'ieee-be');%����λ��һ��ͼ�������
num_colones = fread(fp,1,'int32',0,'ieee-be');%����λ��һ��ͼ�������

images = fread(fp,inf,'unsigned char');%��ȡ�������ݵ�
images = reshape(images,num_rows,num_colones,num_images);%reshape��28x28x60000�ľ����28x28x10000�ľ���
fclose(fp);
