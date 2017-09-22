rounds=2;
anns=cell(rounds,1);
close all;
for i=1:rounds
    fprintf('round %d\n',i );
    %rng(i);
    num_hidden_layer=1;
    num_hidden_neuron=100;
    learning_rate=0.1;
    batch_size=1;
    epoches=200;
    momentum=0.5;
    anns{i} = ANN();
    anns{i}.ANN_load_data();
    anns{i}.set_lumbda(0.00001);
    [train_error,vali_error]=anns{i}.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
    try
        ann=anns{i};
        save(['ann_cross_validation_round' num2str(i+3) '.mat'],'ann');
    catch exception
        fprintf('save round %d failed\n',i);
    end
    
    %filter_plot(anns{i},num_hidden_neuron,i);
end
plot_stats_all(anns);

try
    save('anns_cross_validation_round4_5.mat','anns');
catch exception
    fprintf('final save failed\n');
end




