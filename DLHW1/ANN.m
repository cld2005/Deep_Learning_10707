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
                obj.weights{i}=2*(rand(size_x,size_y)-0.5); % the first layer does not have weights
                obj.biases{i}  =  zeros(obj.layers(i),1);
            end
            
            for i=1:length(obj.layers)
                
                
                obj.preactication{i} = zeros(obj.layers(i),1);
                obj.postactivation{i} = zeros(obj.layers(i),1);
            end
            %fprintf('obj.biases:\n')
            %obj.biases
            
            
        end
        function ANN_load_data (obj)
            [obj.x_train,obj.y_train,obj.x_validate,obj.y_validate,obj.x_test,obj.y_test] = dataLoad();
        end
        
        function [corss_entropy_error, correct]=forward_prop(obj,x,y)
            obj.postactivation{1}=x';

            result = zeros(10,1);
            
            result(int32(y)+1)=int32(1);
            %fprintf('obj.biases inside forward:\n')
            %obj.biases
            for i = 2:obj.num_of_layers
                %fprintf('size of obj.weights{i} %d %d',size(obj.weights{i},1),size(obj.weights{i},2));
                %fprintf('size of postactivation{i-1} %d %d',size(obj.postactivation{i-1},1),size(obj.postactivation{i-1},2));
                %temp1 = obj.weights{i}*transpose(obj.postactivation{i-1});
                %temp2 = obj.biases{i};
                %fprintf('size of temp1 %d %d\n',size(temp1,1),size(temp1,2));
                %fprintf('size of temp2 %d %d\n',size(temp2,1),size(temp2,2));
                obj.preactication{i} = obj.weights{i}*obj.postactivation{i-1}+obj.biases{i};
                %fprintf('obj.postactivation{i}');
                obj.postactivation{i} = transpose(sigmoid(obj.preactication{i}));
            end
            
            obj.output = softmax(obj.postactivation{end});
            
            corss_entropy_error = -1*log(dot(obj.output,result));
            [~,indout]=max( obj.output);
            [~,indres]=max( result);
            correct= (indout==indres);
        end
        function [d_weight, d_bias] = back_prop (obj,y)
            d_weight={};
            d_bias={};
            result = zeros(10,1);
 
            result(int32(y)+1)=int32(1);
            %obj.output
            %result
            grad_out = (obj.output- result).*arrayfun(@d_sigmoid,obj.preactication{end});
            
            for i=(obj.num_of_layers):-1:2%first layer is the input x
                %i
                %fprintf('size of grad_out %d %d\n',size(grad_out,1),size(grad_out,2));
                %fprintf('size of transpose(obj.postactivation{i-1}) %d %d\n',size(transpose(obj.postactivation{i-1}),1),size(transpose(obj.postactivation{i-1}),2));
                d_weight{i}=grad_out*(transpose(obj.postactivation{i-1}));%?????? check 
                d_bias{i}=grad_out;
                grad_h = transpose(obj.weights{i})*grad_out;
                %fprintf('size of obj.preactication{i-1} %d %d\n',size(obj.preactication{i-1},1),size(obj.preactication{i-1},2));
                grad_out=grad_h.*arrayfun(@d_sigmoid,obj.preactication{i-1});
          
            end
            
            
            
            
        end
        

        
        function zero_cell_array = create_new_all_zero (obj,cell_array)
            zero_cell_array={};
            for i =1:size(cell_array,2)
                zero_cell_array{i} = zeros(size(cell_array{i},1),size(cell_array{i},2));
            end
        end
        function [train_error,vali_error] = train(obj,num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum)
        obj.init(num_hidden_layer,num_hidden_neuron);
        train_error=[];
        vali_error=[];
            for epoch = 1:epoches
                fprintf("Epoch %d\n",epoch);
                sample_count=0;
                epoch_cross_entropy_error=0;
                epoch_success_count=0;
                
                batch_d_weight=obj.create_new_all_zero(obj.weights);    
         
                batch_d_bias=obj.create_new_all_zero(obj.biases);
                
                
       

                
                for batch = 0:3000/batch_size-1
                    
                    d_weight=obj.create_new_all_zero(obj.weights);

                    d_bias=obj.create_new_all_zero(obj.biases);
                    %d_bias
            
                    
                    for sub_index = 1:batch_size
                        sample_index = batch*batch_size+sub_index;
                        %fprintf('foward\n');
                        [error_value,correct_count] = obj.forward_prop(obj.x_train(sample_index,:),obj.y_train(sample_index));
                        epoch_cross_entropy_error=epoch_cross_entropy_error+error_value;
                        epoch_success_count=epoch_success_count+correct_count;
                        sample_count=sample_count+1;
                        %fprintf('backward\n');
                        [sub_d_weight,sub_d_bias] = obj.back_prop(obj.y_train(sample_index));
                        
                        d_weight= cellfun(@(c1,c2) c1+c2,d_weight,sub_d_weight,'UniformOutput',false);
                        %d_bias
                        %sub_d_bias
                        d_bias= cellfun(@(c1,c2) c1+c2,d_bias,sub_d_bias,'UniformOutput',false);

                    end
                    
                    d_weight=cellfun(@(c1,c2) momentum*c1+(learning_rate/batch_size)*(c2) ,batch_d_weight,d_weight,'UniformOutput',false);
                    batch_d_weight=d_weight;
                    d_bias=cellfun(@(c1,c2) momentum*c1+(learning_rate/batch_size)*(c2) ,batch_d_bias,d_bias,'UniformOutput',false);
                    batch_d_bias=d_bias;
                    
                    obj.weights = cellfun(@(c1,c2) c1-c2,obj.weights,d_weight,'UniformOutput',false);
                   
                    obj.biases = cellfun(@(c1,c2) c1-c2,obj.biases,d_bias,'UniformOutput',false);

                    
                    ave_error = epoch_cross_entropy_error/sample_count;
                    classification_err_rate = 100*(1-epoch_success_count/sample_count);
                    
                    [validation_err,validation_classification_succ_rate ]=  obj.validate();
                    train_error=vertcat(train_error,[ave_error,classification_err_rate]);
                    vali_error = vertcat(vali_error,[validation_err,validation_classification_succ_rate]);
                    if mod(batch,100)==0
                        fprintf('batch %d\n',batch)
                        fprintf('training cross entropy %f, error rate %f\n',ave_error,classification_err_rate);
                        fprintf('validate cross entropy %f, error rate %f\n',validation_err,100-validation_classification_succ_rate);
                    end
                    
                   
                    
                end % end batch
            end % end epoch
        end % end train
        
        function [corss_entropy_error_rate, correct_rate] = validate(obj)
            corss_entropy_error=zeros(size(obj.x_validate,1),1);
            correct=zeros(size(obj.x_validate,1),1);
            for i=1:size(obj.x_validate,1)
                [e,c]=obj.forward_prop(obj.x_validate(i,:),obj.y_validate(i));
                corss_entropy_error(i)=e;
                correct(i)=c;
            end
            
            %[corss_entropy_error,correct]=arrayfun(@obj.forward_prop,obj.x_validate,obj.y_validate);
            corss_entropy_error_rate= mean(corss_entropy_error);
            correct_rate = mean(correct)*100;

        end
        
        function cell_arry = update_gradiant (obj,c1,c2,momentum,learning_rate,batch_size)
            cell_arry=momentum*c1+learning_rate*(c2/batch_size);
        end

        

        
        
        function y=show_x_train(obj)
            y=obj.x_train;
        end
        
        function y=a(obj,m,n)
            y=m+n;
        end
    end
    
end

