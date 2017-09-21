rounds=4;
anns=cell(rounds,1);
close all;
num_hidden_size=[20,100,200,500];
for i=1:rounds
    fprintf('round %d\n',i );
    %rng(i);
    num_hidden_layer=1;
    num_hidden_neuron=num_hidden_size(i);
    learning_rate=0.01;
    batch_size=1;
    epoches=1;
    momentum=0.5;
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
    save(['ann_hidden_layer' num2str(num_hidden_size(i)) '.mat'],'ann');
end
save('anns_hidden_layer.mat','anns');



