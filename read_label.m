function labels = read_image(filename)
%��ȡ��ǩ��ת��Ϊ60000x1�ľ���
fp = fopen(filename,'rb');
magic = fread(fp,1,'int32',0,'ieee-be');%ħ��
num_items = fread(fp,1,'int32',0,'ieee-be');%��ǩ����
labels = fread(fp,inf,'unsigned char');%��ȡ��ǩ
fclose(fp);