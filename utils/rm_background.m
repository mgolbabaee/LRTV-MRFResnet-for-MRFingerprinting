function [X,mask] = rm_background(X,thresh)
%
% % mask images and remove the background
% =========================================================================
% mask = get_mask(data,par)
%   Input
%       X             data structure of computed quantitative maps X.qmap, and computed proton density X.pd         
%
%       thresh        background masking threshold according to the intensity of pd (default .1)
%
% %
%   Output
%       X              masked quantitative maps
%       mask          [Nx,Ny,(Nz)]     mask  
% =========================================================================
% (c) P. Gomez and M. Golbabaee 2020
% =========================================================================

if nargin<2 
    thresh = 0.1;
end 

data = X.pd;

if ismatrix(data)
    mask = abs(data);
    mask = mask/max(mask(:));
    [x,y] = gradient(mask);
    gradnorm = sqrt(x.^2 + y.^2);
    mask = mask + gradnorm;
elseif ndims(data)==3
    mask = squeeze(sum(abs(data),3)); %sum over extra dims
    mask = mask/max(mask(:));
    [x,y] = gradient(mask);
    gradnorm = sqrt(x.^2 + y.^2);
    mask = mask + gradnorm;
else
    data = data(:,:,:,:);
    mask = squeeze(sum(abs(data),4)); %sum over extra dims
    mask = mask/max(mask(:));
    [x,y] = gradient(mask);
    gradnorm = sqrt(x.^2 + y.^2); %use only 2D gradient
    mask = mask + gradnorm;
end

mask = mask./max(mask(:));
mask = mask>thresh; %mask all data below thresh 


X.qmap = bsxfun(@times, X.qmap, mask);
X.pd = X.pd.*mask;

