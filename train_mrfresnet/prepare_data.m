function [train_x, train_y, nn] = prepare_data(opt)
% This file prepares training samples for the networks,  
% given a MRF dictionary
%
% opt.filename: path to a MRF dictionary
%
% (c) 2018-2020 Mohammad Golbabaee, m.golbabaee@bath.ac.uk
%%
%%
load(opt.filename, 'dict'); % load the dictionary

% scale T1/T2 labels to a range between 0-1 (optional)
label = dict.lut(:,1:2);
nn.scale = 1*max(label,[],1);
label = bsxfun(@rdivide, label, nn.scale); 

nn.V = dict.V; % the low-rank subspace of the MRF dictionary

% Phase-alignment of the MRF dictionary based on the first SVD-compressed
% subspace component
nn.P = angle(dict.D(:,1));
dict.D = bsxfun(@times, dict.D, exp(-1j*nn.P));
dict.D = real(dict.D); % % since the dictionay used to train our example network 
% has only imaginary component, we can first do dephase-compensation and then work/train with real part of the fingerprints signal. 
% (the input to the network are real-valued vectors)

tmp = sqrt(sum(abs(dict.D).^2,2));
dict.D = bsxfun(@times,dict.D, 1./tmp); %make sure dim-reduced data is normalized
nn.normD = dict.normD .* tmp;
train_x = dict.D; 
train_y = label;
