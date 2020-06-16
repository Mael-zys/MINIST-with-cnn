%这里用的是取平均值的池化
function pooled=pool(convolved,pooldim)

con_dim=size(convolved,1);
filter_num=size(convolved,3);
image_num=size(convolved,4);

for i=1:image_num %对i图像
    for j=1:filter_num %对i图像的j个面
        convolve_plan=squeeze(convolved(:,:,j,i));  %获取i图像的j面
        pool_plan=conv2(convolve_plan,ones(pooldim,pooldim)/(pooldim^2),'valid'); %对matrice identite卷积，除以(pooldim^2)，相当于取平均数
        pooled(:,:,j,i)=pool_plan(1:pooldim:con_dim,1:pooldim:con_dim);
    end
end
    