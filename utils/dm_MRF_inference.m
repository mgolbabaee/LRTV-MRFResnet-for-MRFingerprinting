function [out] = dm_MRF_inference(dict, x)
% input 
%        x: Subspace-compressed TSMI (time-series of magnetisation images e.g. computed via LRTV with low-rank subspace dimentionality reduction) 
%        dict: SVD/low-rank compressed MRF dictionary 
%
% output
%       out.qmap: quantitative maps (e.g. T1, T2 maps)
%       out.pd: proton density maps
%
% (c) Mohammad Golbabaee m.golbabaee@bath.ac.uk 2018
%%
disp('performing dictionary matching quantitative inferece...');

recon_dim = size(x); % for 2D Nx,Ny,Timepoints, for 3D Nx,Ny,Nz,Timepoints
x  = reshape(x, [prod(recon_dim(1:end-1)), recon_dim(end)]);
   
%----brture force DM 
for q=size(x,1):-1:1
    Dx = x(q,:) * dict.D';
    [~,idx(q,1)] = max(abs(Dx), [], 2);
end

D = double(dict.D(idx,:));
out.qmap = dict.lut(idx,:);
out.qmap = reshape(out.qmap, [recon_dim(1:end-1), size(dict.lut,2)]);
    
Dx = sum(conj(D) .* x ,2);
Dx  = reshape(Dx,   recon_dim(1:end-1));
out.pd = Dx(:) ./ dict.normD(idx);
out.pd  = reshape(out.pd,   recon_dim(1:end-1));
