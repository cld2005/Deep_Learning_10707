function main()
rounds=1;
anns=cell(5,1);
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
    
    filter_plot(anns{i},num_hidden_neuron);
end

for i=1:rounds
    plot_stats(anns{i}.train_error,anns{i}.vali_error);
end
end

