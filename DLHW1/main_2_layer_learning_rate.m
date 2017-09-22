rounds=4;
anns=cell(rounds,1);
close all;
learning_rates=[0.01 0.1 0.2 0.5];
for i=1:rounds
    fprintf('round %d\n',i );
    %rng(i);
    num_hidden_layer=2;
    num_hidden_neuron=100;
    learning_rate=learning_rates(i);
    batch_size=1;
    epoches=1;
    momentum=0;
    anns{i} = ANN();
    anns{i}.ANN_load_data();
    anns{i}.set_lumbda(0);
    [train_error,vali_error]=anns{i}.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
    ann = anns{i};
    try
        save(['ann_2_layer_learning_rate' num2str(learning_rates(i)) '.mat'],'ann');
    catch exception
        fprintf('save round %d failed\n',i);
    end
    %filter_plot(anns{i},num_hidden_neuron,i);
end
plot_stats_all(anns);

try
    save('anns_2_layer_learning_rate.mat','anns');
catch exception
    fprintf('final save failed\n');
end



