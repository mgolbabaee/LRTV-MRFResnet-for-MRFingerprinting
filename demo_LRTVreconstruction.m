% This code demonstrates a spatiotemporally-regularised convex reconstruction and
% a neural quantitative inference pipeline for solving the Magnetic
% Resonance Fingerprinting (MRF) problem without dictionary matching, according to the paper:
%
%     M. Golbabaee, G. Bounincontri, C. Pirkl, M. Menzel, B. Menze, 
%     M. Davies, and P. Gomez. "Compressive MRI quantification using convex
%     spatiotemporal priors and deep auto-encoders." arXiv preprint arXiv:2001.08746 (2020).
%
% 
% This test example employs a low-dimensional (low-rank) subsapce model 
% as a compact proxy for a larger size MRF dictionary mentioned in this paper.
%
% A toy example Radial k-space trajectory with a with a golden angle increment has been imployed for subspace sampling, 
% similar to J. Asslander's MRF toolbox (https://bitbucket.org/asslaender/nyu_mrf_recon). 
% For this demo, we used the  LR_nufft_operator.m class of the Asslander's toolbox which itself uses Jeff Fessler's nuFFT toolbox (on cpu). 
% This implements an efficient dimension-reduced forward/adjoint operators.
% However if GPU is available, we recommend using NYU's gpunufft
% toolbox for even faster iterations (https://www.opensourceimaging.org/project/gpunufft/)
%
% Key components of this toolbox are first, a convex momentum-accclerated
% algorithm LRTV for solving low-rank and total-variation regularised MRF
% reconstruction problem.
% This algorithm incorporates a 2D or 3D Total Variation (TV) regularisation, in addition 
% to the MRF low-rank subspace constriant [McGivney et al'14, Asslander et al'18], 
% in order to spatiotemporally regularise the MRF inverse problem and
% remove strong aliasing artefacts from TSMIs (time-series of magnetisatioon
% images) in severly undersampled regimes. TV operations use cpu/gpu UNLOCBOX implementations (https://epfl-lts2.github.io/unlocbox-html/).
% The second elemnet of this toolbox is to use Neural network 
% (here a trained encoder-decoder model named MRFResent similar to the paper)
% for fast dictionary-matching-free quantitative inference.
%
% If you find this toolbox useful for preparing your publication, we kindly 
% request citing the following references:
%
% [1] M. Golbabaee, G. Bounincontri, C. Pirkl, M. Menzel, B. Menze, 
%     M. Davies, and P. Gomez. "Compressive MRI quantification using convex
%     spatiotemporal priors and deep auto-encoders." arXiv preprint arXiv:2001.08746 (2020).
%
% [2] J. Asslaender, M.A. Cloos, F. Knoll, D.K. Sodickson, J.Hennig and
%     R. Lattanzi, Low Rank Alternating Direction Method of Multipliers
%     Reconstruction for MR Fingerprinting  Magn. Reson. Med. 2018.
%
% [3] J. A. Fessler and B. P. Sutton, Nonuniform fast fourier
%     transforms using min-max interpolation, IEEE Trans. Signal
%     Process., vol. 51, no. 2, pp. 560-574, Feb. 2003.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Mohammad Golbabaee (m.golbabaee@bath.ac.uk) 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add folder and subfolders in to the path
addpath(genpath([pwd, '/..']));

%% load the MRF dictionary and the corresponding low-rank SVD subspace
% Here we use a dictonary that is already compressed according to [McGivney et al 2014, Asslander et al' 2018] using 10 SVD components. 
% If your dictionary is not compressed, please follow these references and take the SVD decomposition of your dictionary in order to 
% i) determine the low-rank subspace dict.V and ii) compressing your dictionary.
% 
% Data structure:
%   dict.D is the compressed (dimension-reduced) dictionary of normalised fingerprints
%   dict.V is the low-rank subspace of the dictionary 
%   dict.lut is the look up table for T1/T2
%   dict.normD keeps the actual norm of each fingerprint before normalising and saving them in dict.D

load('dictSVD.mat');

%% Constructing a 2D example
% load the brainweb phantom data comprised of: 
%   TSMI: 200x200x880 time-series of magnetisation images with 2D spatial resolution 200x200, and temporal dimension 880 (number of repetitions) 
%   T1_GT, T2_GT, PD_GT: the ground truth T1, T2 and proton density maps used for creating the TSMI phantom 

load('brainweb_phantom');

%% Build a radial trajectory example 
nt = 880; % number of timeframes
Nx = 200; % spatial resolution 200x200 pixels
Ny = Nx;  % spatial resolution 200x200 pixels

GoldenAngle = pi/((sqrt(5.)+1)/2);
kr = pi *  (-1+1/Nx:1/Nx:1).';
phi = (0:nt-1)*GoldenAngle;
k = [];
k(:,2,:) = kr * sin(phi);
k(:,1,:) = kr * cos(phi);

%% Simulate k-space measurements
E = LR_nuFFT_operator(k, [Nx Ny], [], [], 2); % this nuFFT operator is just for sampling (not reconstruction) and it does not incorporate subspace yet

% Sample k-space measurements across temporal frames:
data = E*TSMI;
% Add noise to k-space measuements
rng(1); data = awgn(data, 40);

%% Construct low rank nuFFT Operator [Asslander et al'2018]
% The subspace compression matrix dict.V is used for nufft dimensionality-reduction. It reconstruct dimension-reduced TSMIs through time-domain compression (here from 880 time-frames to 10 SVD-frames)
% The default 5 nearest neighbors are used for interploation with a Kaiser Bessel Kernel and oversampling with a factor of 2 is employed. 
dcomp = col(l2_norm(k,2)); 

% To add an optional density compensation for faster convergence (it improves the conditioning of the forward operator); otherwise, set dcmop to an all-one vector of the same size.
% When we chose to work with dcomp, We need to re-weight the k-space data with sqrt(dcomp))

ELR = LR_nuFFT_operator(k, [Nx Ny], dict.V, [], 2,[],[],dcomp);
data = (data .* sqrt(dcomp)); % density compensated k-space data

%% Reconstruction by DM-free convex LRTV [Golbabaee et al. 2020]
% First, build the Total Variation (TV) operator (here for reconstructing a 2D image slice)
usegpu = 0; % set to 1 for gpu acceleration
TV = TV_operator('2D',usegpu);

% set LRTV optimisation parameters
param.maxit  = 20;
param.step   = 100;  % initial stepsize
param.lambda = 2e-3; % TV regularisation weight
param.backtrack = true; % backtracking adaptive step size search

[TSMI_lrtv] = solve_LRTV(data,ELR,TV,param); % solve LRTV to reconstruct the compressed (dimension-reduced) TSMIs

%% Quantitative inference by MRFResnet [Golbabaee et al. 2020] as compared to Dictionary Matching

load('MRFResnet_AE.mat') % load MRFResnet network 
out_mrfresnet = deep_MRF_inference(nn, TSMI_lrtv); % Compute Neural quantitative inference
out_mrfresnet = rm_background(out_mrfresnet, 0.1); % Remove background where pd intensity is too low


% Try also Dictionary Matching for quantiative inference 
out_dm = dm_MRF_inference(dict, TSMI_lrtv);
out_dm = rm_background(out_dm, 0.1); % Remove background 

%% Reconstruction using non-iterative Filtered back projection i.e. Zero Filling [McGivney et al 2014]
TSMI_zf = ELR' * data; % Filtered back projection (ZF) [McGivney et al 2014]
out_zf = dm_MRF_inference(dict, TSMI_zf); % inference by Dictionary matching 
out_zf = rm_background(out_zf, .3); % Remove background

%% Visualisations

figure(1); % show ground truth
subplot(131); imagesc(T1_GT);axis off; caxis([0, 2]); colormap jet; title('Ground truth: T1');colorbar
subplot(132); imagesc(T2_GT);axis off; caxis([0, .12]); colormap jet; title('Ground truth: T2 ');colorbar
ax(1)=subplot(133); imagesc(abs(PD_GT));axis off; caxis([0, 1]); colormap(ax(1),'pink'); title('Ground truth: PD');colorbar

figure(2); % show LRTV-MRFResnet predicted images and error maps w.r.t. ground-truth
subplot(231); imagesc(out_mrfresnet.qmap(:,:,1));axis off; caxis([0, 2]); colormap jet; title('LRTV-MRFResnet: T1 ');colorbar
subplot(232); imagesc(out_mrfresnet.qmap(:,:,2));axis off; caxis([0, .12]); colormap jet; title('LRTV-MRFResnet: T2 ');colorbar
ax(1)=subplot(233); imagesc(abs(out_mrfresnet.pd));axis off; caxis([0, 1]); colormap(ax(1),'pink'); title('LRTV-MRFResnet: PD ');colorbar
ax(2)=subplot(234); imagesc(out_mrfresnet.qmap(:,:,1)-T1_GT);axis off; caxis([-.1, .1]); colormap(ax(2),'bone'); title('T1 error'); colorbar
ax(3)=subplot(235); imagesc(out_mrfresnet.qmap(:,:,2)-T2_GT);axis off; caxis([-.05, .05]); colormap(ax(3),'bone'); title('T2 error');colorbar
ax(4)=subplot(236); imagesc(abs(out_mrfresnet.pd)-PD_GT);axis off; caxis([-.1, .1]); colormap(ax(4),'bone'); title('PD error');colorbar

figure(3); % show LRTV-DM predicted images and error maps w.r.t. ground-truth
subplot(231); imagesc(out_dm.qmap(:,:,1));axis off; caxis([0, 2]); colormap jet; title('LRTV-MRFResnet: T1 ');colorbar
subplot(232); imagesc(out_dm.qmap(:,:,2));axis off; caxis([0, .12]); colormap jet; title('LRTV-MRFResnet: T2 ');colorbar
ax(1)=subplot(233); imagesc(abs(out_dm.pd));axis off; caxis([0, 1]); colormap(ax(1),'pink'); title('LRTV-MRFResnet: PD ');colorbar
ax(2)=subplot(234); imagesc(out_dm.qmap(:,:,1)-T1_GT);axis off; caxis([-.1, .1]); colormap(ax(2),'bone'); title('T1 error');colorbar
ax(3)=subplot(235); imagesc(out_dm.qmap(:,:,2)-T2_GT);axis off; caxis([-.05, .05]); colormap(ax(3),'bone'); title('T2 error');colorbar
ax(4)=subplot(236); imagesc(abs(out_dm.pd)-PD_GT);axis off; caxis([-.1, .1]); colormap(ax(4),'bone'); title('PD error');colorbar

figure(4); % show ZF-DM predicted images and error maps w.r.t. ground-truth
subplot(221); imagesc(out_zf.qmap(:,:,1));axis off; caxis([0, 2]); colormap jet; title('ZF-DM: T1 ');colorbar
subplot(222); imagesc(out_zf.qmap(:,:,2));axis off; caxis([0, .12]); colormap jet; title('ZF-DM: T2 ');colorbar
%ax(1)=subplot(233); imagesc(abs(out_zf.pd));axis off; caxis([0, 1]); colormap(ax(1),'pink'); title('ZF-DM: PD ');colorbar
ax(2)=subplot(223); imagesc(out_zf.qmap(:,:,1)-T1_GT);axis off; caxis([-.1, .1]); colormap(ax(2),'bone'); title('T1 error');colorbar
ax(3)=subplot(224); imagesc(out_zf.qmap(:,:,2)-T2_GT);axis off; caxis([-.05, .05]); colormap(ax(3),'bone'); title('T2 error');colorbar
%ax(4)=subplot(236); imagesc(abs(out_zf.pd)-PD_GT);axis off; caxis([-.1, .1]); colormap(ax(4),'bone'); title('PD error');colorbar



