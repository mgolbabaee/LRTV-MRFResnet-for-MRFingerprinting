function [train_x, train_y, nn] = prepare_data(opt)
% opt.filename: address to the MRF dictionary
% opt.svd: number of svd components for subspace compression e.g. for an EPG dictionary opt.svd=10
%load(opt.filename)
%%

load(opt.filename, 'dict');

%%

% scale the labels to non-saturating part of sigmoid (e.g. 70%)
label = dict.lut(:,1:2);
nn.scale = 1*max(label,[],1);
label = bsxfun(@rdivide, label, nn.scale); 
% 
% % FISP data is imag only (for more general case comment this line)
% dict.D = imag(dict.D(:,1:opt.cutoff)); 
% 
% dict.D = bsxfun(@times, dict.D, 1./sqrt(sum(abs(dict.D).^2,2)));
% 
% % linear dim reduction (10 svd is enough for this data)   
% [~,~,V] = svd(dict.D.'*dict.D); V=V(:,1:opt.svd);
% dict.D= dict.D*V; 

nn.V = dict.V(:,1:opt.svd);
dict.D = dict.D(:,1:opt.svd);

%dephase dictionary (based on 1st eigen basis)
nn.P = angle(dict.D(:,1));
dict.D = bsxfun(@times, dict.D, exp(-1j*nn.P));
dict.D = real(dict.D); % if dephasing is perfect then imag part is always zero    

tmp = sqrt(sum(abs(dict.D).^2,2));
dict.D = bsxfun(@times,dict.D, 1./tmp); %make sure dim-reduced data is normalized
% [nSmp, nDim] = size(data);
nn.normD = dict.normD .* tmp;
train_x = dict.D; 
train_y = label;