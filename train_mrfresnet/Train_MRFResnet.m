% The main file/script to train a Encoder-Decoder neural network MRFResnet 
% in order to replace dictionary matching in Magnetic Resonance Fingerprinting problem
%
%
% (c) 2018-2020 Mohammad Golbabaee, m.golbabaee@bath.ac.uk
%%
clear; clc;

%% Data preparation
opt.filename = '../data_models/dictSVD';
[train_x,train_y, nn] = prepare_data(opt);

%% Noisy data augmentation for training the Encoder nn.Netfw
% Add noise to each fingerprint and then do dictionary search to see to
% which T1/T2 labels it matches best, then do the label correction.

opt.augsize = 10; % number of noisy realisations of each figerprint (50 in the paper).

X=[]; Y=[];
for i =1:opt.augsize
    C=ceil(size(train_x,1)/4);
    for j=1:4
        B=[(j-1)*C+1: min( (j)*C, size(train_x,1) )];
        
        batch_x = train_x(B,:)+ 1e-2*randn(size(train_x(B,:))); % added noise
        
        batch_x = bsxfun(@times,batch_x, 1./sqrt(sum(abs(batch_x).^2,2))); %normalize batch       
        [~,ind] = max( real(train_x*batch_x') ,[],1); % do the dictionary search
        batch_y = train_y(ind,:); % correct labels according to  search result
        X = [X;batch_x];
        Y =[Y;batch_y];
    end
end

%% standardise/normalise traing samples
[X, tr_mu, sigma]= zscore(X);
nn.sigma=sigma;
nn.tr_mu=tr_mu;

X = single(reshape(X.',1,1, size(X,2), size(X,1)));
Y = reshape(Y.',1,1,size(Y,2),size(Y,1));
%% Train the Encoder model nn.Netbw
% This is residual network MRFResnet which learns to embedd Dictionary-Matching and 
% find correct (T1,T2) values given noisy fingerprints. 
modelname = 'Encoder';
nn.NetFw = model_trainer(X,Y,modelname);

%% Train the Decoder model nn.bw
% This is a simple shallow network with one hiden layer which creates clean 
% magnetic responses (fingerprints) given (T1,T2)

% prepare training data 
X = train_y;
Y = bsxfun(@times, train_x ,nn.normD); % We want to estimate the true (un-normalizxed) Bloch responses
X = single(reshape(X.',1,1, size(X,2), size(X,1)));
Y = reshape(Y.',1,1,size(Y,2),size(Y,1));

% train
modelname = 'Decoder';
nn.NetBw = model_trainer(X,Y,modelname);

%% save results
filename = [opt.filename,'_MRFResnet_AE',datestr(now),'.mat'];
save(filename,'nn')