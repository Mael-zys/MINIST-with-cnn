function images = read_image(filename)
%读取像素点，转化为28x28x60000的矩阵或28x28x10000的矩阵
fp = fopen(filename,'rb');
magic = fread(fp,1,'int32',0,'ieee-be');%文件第一位是魔数
num_images = fread(fp,1,'int32',0,'ieee-be');%第二位是图像个数
num_rows = fread(fp,1,'int32',0,'ieee-be');%第三位是一个图像的行数
num_colones = fread(fp,1,'int32',0,'ieee-be');%第四位是一个图像的列数

images = fread(fp,inf,'unsigned char');%读取所有数据点
images = reshape(images,num_rows,num_colones,num_images);%reshape成28x28x60000的矩阵或28x28x10000的矩阵
fclose(fp);
