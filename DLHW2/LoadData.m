function [ train,validate,test] = LoadData()
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    fprintf('reading training data ...\n')
    tr_data=textread('../DLHW1/digitstrain.txt','','delimiter',',');
    tr_data = tr_data(randperm(size(tr_data,1)),:);%shuffle data
    train=tr_data;
    
    
    fprintf('reading validation data ...\n')
    va_data=textread('../DLHW1/digitsvalid.txt','','delimiter',',');
    validate=va_data;

    fprintf('reading test data ...\n')
    ts_data=textread('../DLHW1/digitstest.txt','','delimiter',',');
    test=ts_data;

end

