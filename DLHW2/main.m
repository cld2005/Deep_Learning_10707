close all;
rounds=1;
rbms=cell(rounds,1);
for i=1:rounds
    fprintf('round %d\n',i );

    num_visible=784;
    num_hiddenn=100;
    learning_rate=0.1;
    batch_size=1;
    epoches=3;
    k=1;
    rbms{i} = RBM();
    [train_error,vali_error] = rbms{i}.train(num_visible,num_hiddenn,learning_rate,batch_size,epoches,k);
    filter_plot(rbms{1},100,i);
end

plot_stats_all(rbms);