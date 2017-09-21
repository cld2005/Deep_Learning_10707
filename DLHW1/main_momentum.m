rounds=3;
anns=cell(rounds,1);
close all;
momentums=[0,0.5,0.9];
for i=1:rounds
    fprintf('round %d\n',i );
    %rng(i);
    num_hidden_layer=1;
    num_hidden_neuron=100;
    learning_rate=0.1;
    batch_size=1;
    epoches=200;
    momentum=momentums(i);
    anns{i} = ANN();
    anns{i}.ANN_load_data();
    [train_error,vali_error]=anns{i}.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
    
    %filter_plot(anns{i},num_hidden_neuron,i);
end
plot_stats_all(anns);
%for i=1:rounds
    %plot_stats(anns{i}.train_error,anns{i}.vali_error);
%end
for i=1:size(anns,1)
    ann = anns{i};
    save(['ann_momentum' num2str(momentums(i)) '.mat'],'ann');
end
save('ann_momentum.mat','anns');



