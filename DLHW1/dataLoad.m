function [x_train,y_train,x_validate,y_validate,x_test,y_test] = dataLoad()
fprintf('exist x_train %d ...\n',exist('x_train'))
if(exist('x_train')==0)
    fprintf('reading xtrain ...\n')
    tr_data=textread('digitstrain.txt','','delimiter',',');
    x_train=tr_data(:,1:784);
    y_train=tr_data(:,785);
end
if(exist('x_validate')==0)
    va_data=textread('digitsvalid.txt','','delimiter',',');
    x_validate=va_data(:,1:784);
    y_validate=va_data(:,785);   
end

if(exist('x_test')==0)
    ts_data=textread('digitstest.txt','','delimiter',',');
    x_test=ts_data(:,1:784);
    y_test=ts_data(:,785);
end
end
