function x = l2_norm(x, dim, squeeze_flag)
% Calculates the l2 norm either over the entire array or along any set of
% dimesions
%
% x = l2_norm(x)
% x = l2_norm(x, dim)
% x = l2_norm(x, dim, squeeze_flag)
%
% Input:
%   x            = multi dimensional array of data
%   dim          = vector of dimensions over which the norm is calculated
%   squeeze_flag = Boolean indicating whether the final result shall be
%                  squeezed. Default = 1
%
% Example:
%   x = l2_norm(rand(256,256,4), [1 3]);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Jakob Asslaender, August 2016
% New York University School of Medicine, Center for Biomedical Imaging
% University Medical Center Freiburg, Medical Physics
% jakob.asslaender@nyumc.org
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 2 || isempty(dim)
    dim=1:length(size(x));
end

for j=1:length(dim)
        x = sqrt(sum(abs(x).^2,dim(j)));
end

if nargin<3 || isempty(squeeze_flag) || squeeze_flag == 1
    x = squeeze(x);
end