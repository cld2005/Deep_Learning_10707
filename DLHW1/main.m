num_hidden_layer=1;
num_hidden_neuron=100;
learning_rate=0.1;
batch_size=20;
epoches=100;
momentum=0.5;
ann = ANN();
%ann.init(num_hidden_layer,num_hidden_neuron);
ann.ANN_load_data();
[train_error,vali_error]=ann.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);


