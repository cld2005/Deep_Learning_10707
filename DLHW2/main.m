close all;
rmb = RBM();
num_visible=784;
num_hiddenn=100;
learning_rate=0.1;
batch_size=1;
epoches=200;
k=1;
[train_error,vali_error] = rmb.train(num_visible,num_hiddenn,learning_rate,batch_size,epoches,k);