function layers = RSB(datach, nbch2,depth,tag)

% (c) 2018-2020 Mohammad Golbabaee, m.golbabaee@bath.ac.uk
%%
layers = [];
for i = 1:depth
    layers = [layers
        convolution2dLayer(1, nbch2,'Padding',0, 'Name', [tag, '_conv1',num2str(i)])
        %batchNormalizationLayer('Name',[tag, '_BN',num2str(i)]);
        reluLayer('Name',[tag, num2str(i),'relu'])];
end
layers = [layers 
    convolution2dLayer(1, datach,'Padding',0, 'Name',[tag,'_conv2',num2str(i)]);
    %batchNormalizationLayer('Name',[tag,'_BN',num2str(i+1)]);
    additionLayer(2,'Name',[tag,'_add'])];

end