classdef ANN < handle
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        layers = [];
        num_of_layers = 0;
        biases={};
        weights={}; % for 2 hidden layers 100*784/100*100/10*100
        preactication={}; % for 2 hidden layers 784/100/100/10
        postactivation={}; % for 2 hidden layers 784/100/100/10
        output=[];
        x_train=[];
        y_train=[];
        x_validate=[];
        y_validate=[];
        x_test=[];
        y_test=[];
    
    end
    
    methods
        function init(obj,num_hidden_layer,num_hidden_neuron)
            obj.num_of_layers = num_hidden_layer+2;
            obj.layers(end+1)=784;% input layer is always 784
            for i=1:num_hidden_layer
                 obj.layers(end+1)=num_hidden_neuron; % add hidden layers
            end
            obj.layers(end+1)=10; % output layer
            
            weigits_size = horzcat(transpose(obj.layers(2:end)),transpose(obj.layers(1:end-1)));
            
            for i=2:obj.num_of_layers
                size_x = weigits_size(i-1,1);
                size_y =weigits_size(i-1,2);
                obj.weights{i}=rand(size_x,size_y); % the first layer does not have weights
            end
            
            for i=1:length(obj.layers)
                obj.biases{i}  =  zeros(obj.layers(i),1);
                obj.preactication{i} = zeros(obj.layers(i),1);
                obj.postactivation{i} = zeros(obj.layers(i),1);
            end
            
        end
        function ANN_load_data (obj)
            [obj.x_train,obj.y_train,obj.x_validate,obj.y_validate,obj.x_test,obj.y_test] = dataLoad();
        end
        
        function [corss_entropy_error, correct]=forward_prop(obj,x,y)
            obj.postactivation{1}=x;
            result = zeros(10,1);
            result(y)=1;
            
            for i = 2:obj.num_of_layers
                obj.preactlayer{i} = dot(obj.weigths{i},obj.postactivation{i-1})+obj.biases(i);
                obj.postactivation{i} = sigmoid(obj.preactlayer{i});
            end
            
            obj.output = softmax(obj.postactivation{end});
            
            corss_entropy_error = -1*log(dot(obj.output,result'));
            [~,indout]=max( obj.output);
            [~,indres]=max( result);
            correct= (indout==indres);
        end
        
        function [d_weight, d_bias] = back_prop (obj,y)
            d_weight={};
            d_bias={};
            result = zeros(10,1);
            result(y)=1;
            
            grad_out = obj.output- result(y);
            
            for i=(obj.num_of_layers):-1:2%first layer is the input x
                
                d_weight{i}=dot(grad_out,transpose(obj.postactivation{i-1}));%?????? check 
                d_bias{i}=grad_out;
                grad_h = dot(transpose(obj.weigths{i}),grad_out);
                grad_out=grad_h.*d_sigmoid(obj.preactication{i-1});
          
            end
            
            
            
            
        end
        
        function [train_error,vali_error] = train(obj,num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum)
        obj.init(num_hidden_layer,num_hidden_neuron);
        train_error=[];
        vali_error=[];
            for epoch = 1:epoches
                batch_d_weight={};
                for i=1:obj.num_of_layers
                    batch_d_weight{i} = zeros(size(obj.weigths{i},1),size(obj.weigths{i},2));
                end 
                batch_d_bias={};

                for i =1:obj.num_of_layers
                    batch_d_bias = zeros(size(obj.biases{i},1),size(obj.biases{i},2));
                end

                for batch =1:batch_size
                    d_weight={}
                    for i=1:obj.num_of_layers
                        d_weight{i} = zeros(size(obj.weigths{i},1),size(obj.weigths{i},2));
                    end 

                    d_bias={};
                    for i =1:obj.num_of_layers
                        d_bias{i} = zeros(size(obj.biases{i},1),size(obj.biases{i},2));
                    end



                end
            end
        end

        

        
        
        function y=show_x_train(obj)
            y=obj.x_train;
        end
    end
    
end

