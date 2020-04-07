# LRTV-MRFResnet 

This toolbox provides codes for solving Magnetic Resonance Fingerprinting (MRF) problems 
without dictionary matching through a spatiotemporally-regularised convex reconstruction and
neural quantitative inference according to the paper:

    M. Golbabaee, G. Bounincontri, C. Pirkl, M. Menzel, B. Menze, 
    M. Davies, and P. Gomez. "Compressive MRI quantification using convex
    spatiotemporal priors and deep auto-encoders." arXiv preprint arXiv:2001.08746 (2020).

Key components of this toolbox are:
1- The convex and momentum-acclerated algorithm solve_LRTV.m for solving low-rank and total-variation regularised MRF reconstruction problem.
2- The encoder-decoder network (MRFResent_AE.mat) for fast quantitative inference. Codes for training this model are in the folder train_mrfresnet.

The LRTV algorithm incorporates a 2D or 3D Total Variation (TV) regularisation, in addition 
to the MRF low-rank subspace constriant [Asslander et al'18], in order to spatiotemporally regularise the MRF inverse problem and remove strong aliasing artefacts from the TSMIs (time-series of magnetisatioon images) in severly undersampled regimes. 

TV operations use cpu/gpu UNLOCBOX implementations (https://epfl-lts2.github.io/unlocbox-html/).
For this demo, we used the LR_nufft_operator.m class of [Asslander et al'18] which itself uses Fessler's nuFFT toolbox.
This implements an efficient dimension-reduced forward/adjoint operators. However if GPU is available, we recommend using NYU's gpunufft toolbox for even faster iterations (https://www.opensourceimaging.org/project/gpunufft/)
 

If you find this toolbox useful for preparing your publication, we kindly 
request citing the following references:

[1] M. Golbabaee, G. Bounincontri, C. Pirkl, M. Menzel, B. Menze, 
    M. Davies, and P. Gomez. "Compressive MRI quantification using convex
    spatiotemporal priors and deep auto-encoders." arXiv preprint arXiv:2001.08746 (2020).

[2] J. Asslaender, M.A. Cloos, F. Knoll, D.K. Sodickson, J.Hennig and
    R. Lattanzi, Low Rank Alternating Direction Method of Multipliers
    Reconstruction for MR Fingerprinting  Magn. Reson. Med. 2018.

[3] J. A. Fessler and B. P. Sutton, Nonuniform fast fourier
    transforms using min-max interpolation, IEEE Trans. Signal
    Process., vol. 51, no. 2, pp. 560-574, Feb. 2003.

Tested on MATLAB_R2019a

(c) Mohammad Golbabaee
