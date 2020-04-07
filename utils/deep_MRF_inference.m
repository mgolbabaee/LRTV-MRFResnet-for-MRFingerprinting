function [out] = deep_MRF_inference(nn,X)
% input 
%        x: Subspace-compressed TSMI (time-series of magnetisation images e.g. computed via LRTV with low-rank subspace dimentionality reduction) 
%        nn: an auto-encoder network which embeds Dictionary Matching 
%
%               nn.fw: a Decoder network such as MRFResnet to estimate T1/T2 maps given noisy magnetisation responses
%               nn.bw: an Encoder network to generate clean magnetisation responses, given the T1/T2 parameters
%
% output
%       out.qmap: quantitative maps (e.g. T1, T2 maps)
%       out.pd: proton density maps

% (c) Mohammad Golbabaee m.golbabaee at bath.ac.uk 2018
%%
disp('performing neural quantitative inferece...');
%Note: if data is too large (e.g. in 3D reconstruction) to be fed at once to the network, chop it down to minibatches and then feed it to the network stagewise

dim = size(X);
X = transpose(reshape(X,[],dim(end)));

% Phase alignment of the inputs
tmp_n=sqrt(sum(abs(X).^2,1));
P = angle(X(1,:));
X = bsxfun(@times, X, 1./((tmp_n+1e-8).*exp(1j*P)));
X = real(X); % since the dictionay used to train our example network has only imaginary component, we can first do dephase-compensation and then work with real signal part. (similarly network was trained using phase-compensated fingerprints)

%% parameter estimation (here T1,T2) using the Encoder network nn.fw

X = reshape(X,1,1, size(X,1), size(X,2));

if isfield(nn,'tr_mu') % load data normalisation/standardisation parameters: mu = mean, sigma = std.
    tr_mu=reshape(nn.tr_mu,1,1,dim(end),1);
    sigma=reshape(nn.sigma,1,1,dim(end),1);
end

maps= nn.NetFw.predict( (X-tr_mu)./sigma);
maps = reshape(maps.',1,1, size(maps,2), size(maps,1));

%% Magnetic response estimation using the Decoder network nn.bw

D_atom = (nn.NetBw.predict(maps)).';

maps = (reshape(maps, [],size(maps,4))).';
maps = bsxfun(@times,maps, nn.scale);
maps=reshape(maps, [dim(1:end-1), 2]);
X = reshape(X,[], size(X,4));

%% computing the proton density
coeff = sum(X.*conj(D_atom),1)./sum(abs(D_atom).^2,1);
coeff = coeff.*tmp_n.*exp(1j*P);
coeff=reshape(coeff,[dim(1:end-1)]);

%%
out.qmap = maps;
out.pd = coeff;
end

