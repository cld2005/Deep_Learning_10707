rounds=5;
anns=cell(5);
for i=1:5
    fprintf('round %d\n',i );
    rng(i);
    num_hidden_layer=1;
    num_hidden_neuron=100;
    learning_rate=0.1;
    batch_size=10;
    epoches=201;
    momentum=0;
    anns{i} = ANN();
    anns{i}.ANN_load_data();
    [train_error,vali_error]=anns{i}.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
    plot_stats(train_error,vali_error);
    filter_plot(anns{i},num_hidden_neuron);
end

