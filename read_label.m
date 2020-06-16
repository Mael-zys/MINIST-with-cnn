function labels = read_image(filename)
%读取标签，转化为60000x1的矩阵
fp = fopen(filename,'rb');
magic = fread(fp,1,'int32',0,'ieee-be');%魔数
num_items = fread(fp,1,'int32',0,'ieee-be');%标签个数
labels = fread(fp,inf,'unsigned char');%读取标签
fclose(fp);