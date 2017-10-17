classdef RBM  < handle
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        x_train=[];
        x_validate=[];
        x_test=[];
        num_visible;
        num_hiddenn;
        weights=[];
        bias_vh=[];
        bias_hv=[];
        
        
    end
    
    methods
        function RBM_load_data(obj)
            [obj.x_train,obj.x_validate,obj.x_test]=LoadData();
        end
        function init(obj,num_visible,num_hiddenn)
            obj.num_hiddenn=num_hiddenn;
            obj.num_visible=num_visible;
            obj.RBM_load_data();
            %obj.weights=rand(obj.num_visible,obj.num_hiddenn);
            obj.weights=normrnd(0,1,[obj.num_hiddenn obj.num_visible ]);% 100*784 initialize to random gaussian
            obj.bias_vh=zeros(1,obj.num_hiddenn);%1*100
            obj.bias_hv=zeros(1,obj.num_visible);%1*784
        end
        function [train_error,vali_error] = train(obj,num_visible,num_hiddenn,learning_rate,batch_size,epoches,k)
            obj.init(num_visible,num_hiddenn);
            train_error=zeros(epoches,1);
            vali_error=zeros(epoches,1);
            
            
            for epoch = 1:epoches
                fprintf('Epoch %d\n',epoch);
                for batch=1:floor(3000/batch_size)-1
                    start_bond  = 1+(batch-1)*30;
                    end_bond=batch*30;
                    batch_data = obj.x_train(start_bond:end_bond,:);
                    positive_v = batch_data;
                    positive_h = obj.h_given_v(positive_v);
                    
                    negative_v=positive_v;
                    negative_h = positive_h;
                    k=max(1,k);
                    for i=1:k
                        negative_v = obj.v_given_h(negative_h);
                        negative_h = obj.h_given_v(negative_v);
                    end
                    
                    d_weights = (positive_h'*positive_v- negative_h'*negative_v);
                    d_bias_vh = mean(positive_h-negative_h);
                    d_bias_hv = mean (positive_v-negative_v);
                    
                    obj.weights = obj.weights+ learning_rate*d_weights;
                    obj.bias_hv =  obj.bias_hv +learning_rate*d_bias_hv;
                    obj.bias_vh =  obj.bias_vh +learning_rate*d_bias_vh;
                    
                end
                train_error(epoch,1)= obj.cal_cross_entropy(obj.x_train);
                vali_error(epoch,1) = obj.cal_cross_entropy(obj.x_validate);
                
               fprintf('training cross entropy %f\n',train_error(epoch,1));
               fprintf('validate cross entropy %f\n',vali_error(epoch,1));

            end
        end
        
        function h = h_given_v(obj,v)
            h= v*transpose(obj.weights)+obj.bias_vh;
            h = arrayfun(@(x) sigmoid(x),h);
        end
        
        function v = v_given_h(obj,h)
            v = h*obj.weights+obj.bias_hv;
            v = arrayfun(@(x) sigmoid(x),v);
        end
        
        function cross_entropy=cal_cross_entropy(obj,positive_v)
            h = positive_v*transpose(obj.weights)+obj.bias_vh;
            h = arrayfun(@(x) sigmoid(x),h);
            
            v = h*obj.weights+obj.bias_hv;
            v = arrayfun(@(x) sigmoid(x),v);
            cross_entropy = -mean (sum (dot(positive_v,log(v))+dot((1-positive_v),log(1-v))));
        end

    end
    
end

