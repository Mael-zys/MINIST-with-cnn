%�����õ���ȡƽ��ֵ�ĳػ�
function pooled=pool(convolved,pooldim)

con_dim=size(convolved,1);
filter_num=size(convolved,3);
image_num=size(convolved,4);

for i=1:image_num %��iͼ��
    for j=1:filter_num %��iͼ���j����
        convolve_plan=squeeze(convolved(:,:,j,i));  %��ȡiͼ���j��
        pool_plan=conv2(convolve_plan,ones(pooldim,pooldim)/(pooldim^2),'valid'); %��matrice identite���������(pooldim^2)���൱��ȡƽ����
        pooled(:,:,j,i)=pool_plan(1:pooldim:con_dim,1:pooldim:con_dim);
    end
end
    