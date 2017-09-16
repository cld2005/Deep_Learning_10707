classdef ANN < handle
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        layers = [];
        num_of_layers = 0;
        biases=[];
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
        function init(obj,num_hidden_layer)
            obj.num_of_layers = num_hidden_layer+2;
            obj.layers(end+1)=748;% input layer is always 748
            for i=1:num_hidden_layer
                 obj.layers(end+1)=100; % add hidden layers
            end
            obj.layers(end+1)=10; % output layer
            
            weigits_size = horzcat(transpose(obj.layers(2:end)),transpose(obj.layers(1:end-1)));
            
            for i=2:obj.num_of_layers
                size_x = size(weigits_size,1);
                size_y = size(weigits_size,2);
                obj.weights{i}=rand(size_x,size_y);
                
            end
            
        end
        function ANN_load_data (obj)
            [obj.x_train,obj.y_train,obj.x_validate,obj.y_validate,obj.x_test,obj.y_test] = dataLoad();
        end
        
        function forward_prop(obj,x)
            obj.postactivation{1}=x;
            
            for i = 2:obj.num_of_layers
            
            end
        end
        
        function y=show_x_train(obj)
            y=obj.x_train;
        end
    end
    
end

