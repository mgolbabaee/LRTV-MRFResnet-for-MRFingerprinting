function [J]= TV_operator(mode,usegpu)
% This function build Total Variation (TV) operators using Unlocx toolbox
%
% Input
%   mode: '2D' or '3D' for 2D or 3D Total variation regularisation 
%   use gpu: 1 for gpu computation, 0 for cpu computation
%
% output
% operator J 
%           J.prox: proximal TV shrinkage operator (shrinkage parameter gamma TBD by user)
%           J.norm: TV norm
%
% Note: For 2D, the TSMI has dimensions [Nx_pixels,Ny_pixels,Timepoints], For 3D, the TSMI has dimensions [Nx_pixels,Ny_pixels, Nz_pixels,Timepoints],
%
% (c) Mohammad Golbabaee (m.golbabaee@bath.ac.uk) 2019
%% =========== 

switch mode
    case '3D' %3D TV
J.norm = @(x) NORMTV3D(x);  
J.prox = @(x,gamma) proxTV3D(x,gamma,usegpu);
    case '2D' %2D TV
J.norm = @(x) NORMTV2D(x);  
J.prox = @(x,gamma) proxTV2D(x,gamma,usegpu);
end

end


function x2 = proxTV3D(x2,gamma,usegpu)

dim = size(x2);

paramtv.useGPU=usegpu;
paramtv.verbose = 0;
for i=1:dim(4)
x2(:,:,:,i) = prox_tv3d(x2(:,:,:,i), gamma, paramtv); 
end

end


function N = NORMTV3D(x2)

dim = size(x2);

N=0;
for i=1:dim(4)
    N=N+norm_tv3d(x2(:,:,:,i));
end

end

function x2=proxTV2D(x2,gamma,usegpu)
dim = size(x2);

paramtv.useGPU=usegpu;
paramtv.verbose = 0;
for i=1:dim(3)
x2(:,:,i) = prox_tv(squeeze(x2(:,:,i)), gamma, paramtv); 
end
end

function N=NORMTV2D(x2)
dim = size(x2);

N=0;
for i=1:dim(3)
    N=N+norm_tv(x2(:,:,i));
end

end