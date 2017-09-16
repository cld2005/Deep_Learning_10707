classdef ANN < handle
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        layers = [];
        num_of_layers = 0;
        biases=[];
        weights={};
        preactication=[];
        postactivation=[];
        output=[];
        x_train=[];
        y_train=[];
        x_validate=[];
        y_validate=[];
        x_test=[];
        y_test=[];
    
    end
    
    methods
        function ANN_load_data (obj)
            [obj.x_train,obj.y_train,obj.x_validate,obj.y_validate,obj.x_test,obj.y_test] = dataLoad();
        end
        
        function y=show_x_train(obj)
            y=obj.x_train;
        end
    end
    
end

