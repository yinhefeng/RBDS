clear
clc
close all

load('D:\test\AR_55x40.mat')

maxValue=max(DATA(:));
DATA=DATA/maxValue;

EachClassNum=26;
ClassNum=length(unique(Label));
experiments=3;
basePerCls=5;

param=[];
param.alpha=1e-2;
param.lambda=1e-2;
param.beta=1e-2;
param.gamma=0.05;
param.clsNum=ClassNum;

param.dicNumPerCls=basePerCls;
reg_rate=zeros(1,experiments);

for ii=1:experiments
    ii
    train_ind=[];
    test_ind=[];
    for cls_ind=1:ClassNum
        temp=zeros(1,EachClassNum);
        temp(1:7)=1;
        randnum=randperm(3);%3 occluded images by sunglass for each subject in S1
        temp(randnum(1)+7)=1;
        train_ind=[train_ind,temp];
        
        temp1=zeros(1,EachClassNum);
        temp1(14:23)=1;
        if randnum(1)==1
            temp1(:,[9,10])=1;
        elseif randnum(1)==2
            temp1(:,[8,10])=1;
        elseif randnum(1)==3
            temp1(:,[8,9])=1;
        end
        test_ind=[test_ind,temp1];
    end
    
    train_ind=logical(train_ind);
    test_ind=logical(test_ind);
    
    train_data=DATA(:,train_ind);
    train_label=Label(:,train_ind);
    H_train = full(ind2vec(train_label,ClassNum));
    
    test_data=DATA(:,test_ind);
    test_label=Label(:,test_ind);
    
    train_tol=length(train_label);
    test_tol=length(test_label);
    
    param.trainNumPerCls=sum(1==train_label);
    
    %     rand_num=randperm(param.trainNumPerCls*ClassNum);
    %     ind=rand_num(1:atom_tol);
    %     D=train_data(:,ind);
    
    train_num=param.trainNumPerCls;
    D_ind=[];
    for i=1:ClassNum
        temp=zeros(1,train_num);
        randnum=randperm(train_num);
        temp(randnum(1:basePerCls))=1;
        D_ind=[D_ind,temp];
    end
    D_ind=logical(D_ind);
    D = train_data(:,D_ind);
    
    [Z_tr,D,E] =  BDLRR(train_data, D, param);
    
    Z_tt = lrsr(test_data,D,param);
    
    lambda = 0.1;
    W = H_train*Z_tr'/(Z_tr*Z_tr'+lambda*eye(size(Z_tr*Z_tr')));
    [~,pre_label] = max(W*Z_tt);
    sum(pre_label==test_label)/test_tol
    
    reg_rate(ii) = sum(pre_label==test_label)/test_tol
end

mean(reg_rate)
std(reg_rate)