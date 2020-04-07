classdef LR_nuFFT_operator
    % LR_nuFFT_operator performs a transformation from a low rank image
    % space to the non-Cartesian k-space trajectory in the temporal domain.
    % Alternatively, if the matrix u is not specified, the operator
    % performs a frame-frame nuFFT - or even if Nt=1, just a nuFFT of a
    % single frame.
    %
    % A = LR_nuFFT_operator(trajectory, imageDim, u, smaps, os, neighbors, kernel, dcomp)
    %
    % Input:
    %   trajectory = [N_samples 2(3) Nt] (obligatory)
    %                Arbitrary trajectory in 2D or 3D k-space N_samples is
    %                the lenth of the trajectory for each time frame, 2(3)
    %                depends on 2D or 3D imaging and Nt is the number of
    %                time frames to be transformed.
    %                k-space is defined in the range -pi - pi
    %
    %     imageDim = [1 2(3)] (obligatory)
    %                [Nx Ny (Nz)]: Spatial dimensions of the image.
    %
    %            u = [Nt R] (optional)
    %                Matrix that transform the temporal domain into a
    %                domain of rank R. If left emtpy, a frame-by-frame
    %                nuFFT is performed.
    %
    %        smaps = [Nx Ny (Nz) Ncoils] (optional)
    %                Coil sensitivity maps.
    %
    %           os = Scalar: Oversampling (optional; default = 1)
    %
    %    neighbors = [1 2(3)] (optional; default 5 in each dimension)
    %                [x_neighbors y_neighbors (z_neighbors)]
    %                Number of neighbors used for interpolation;
    %
    %       kernel = (optional; default = 'kaiser')
    %                'kaiser' for Kaiser-Bessel interpolation or
    %                'minmax:kb' for Fessler's Min-Max kernel with Kaiser-Bessel
    %                based scaling;
    %
    %        dcomp = [N_samples Nt] (optional)
    %                Density compensation; The square root of it is applied
    %                during the forward and adjoint operation in order to
    %                fulfill the requirements of the adjoint. Consequently,
    %                the data must be pre-multiplied by the square root of
    %                the density compensation before calling A'*x.
    %
    % Output: Object
    %
    % This operator as acts similar to a matrix. The actual nuFFT is
    % implemented by the mtimes function, which is performed by calling
    % something like
    %
    % x = rand(Nx,Ny,Nz,R);
    % b = A * x;
    %
    % And the adjoint operation is called by something like
    %
    % b = rand(N_samples, Nt, Ncoils);
    % x = A' * b;
    %
    % This operator employs Jeff Fesslers nufft. Please cite
    %   J. A. Fessler and B. P. Sutton, Nonuniform fast fourier
    %   transforms using min-max interpolation, IEEE Trans. Signal
    %   Process., vol. 51, no. 2, pp. 560?574, Feb. 2003.
    %
    % More details can be found in
    %   J. Asslaender, M.A. Cloos, F. Knoll, D.K. Sodickson, J.Hennig and
    %   R. Lattanzi, Low Rank Alternating Direction Method of Multipliers
    %   Reconstruction for MR Fingerprinting  Magn. Reson. Med., epub
    %   ahead of print, 2016.
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (c) Jakob Asslaender, August 2016
    % New York University School of Medicine, Center for Biomedical Imaging
    % University Medical Center Freiburg, Medical Physics
    % jakob.asslaender@nyumc.org
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties
        numCoils = [];            % Number of receive coils
        imageDim = [];            % Image dimensions [Nx Ny (Nz) R]
        imageDimOS = [];          % Image dimensions incl. oversampling
        adjoint = false;          % Boolean to indicate the adjoint
        trajectory_length = [];   % Length of the k-space trajectory
        nuFFTNeighbors = [];      % Number of neighbors used for interpolation
        smaps = {};               % Sensititvity profiles for SENSE
        p = [];                   % Sparse Matrix which implements the gridding
        sn = [];                  % Scaling for better nuFFT performance
        dcomp = [];               % Density compensation
    end
    
    methods
        function  A = LR_nuFFT_operator(trajectory, imageDim, u, smaps, os, neighbors, kernel, dcomp)
            % Constructor
            
            if nargin > 2 && ~isempty(u)
                u = double(u);
                imageDim(end+1) = size(u,2);          % Add rank
            else
                imageDim(end+1) = size(trajectory,3); % Add nt     
            end
            
            
            if nargin<=3 || isempty(smaps) % Without SENSE
                A.numCoils = 1;
                A.smaps = 1;
            else                           % With SENSE
                % Get number of coils
                if size(trajectory,2) == 3 && length(size(smaps))== 4
                    A.numCoils = size(smaps, length(size(smaps)));
                end
                if size(trajectory,2) == 3 && length(size(smaps))== 3
                    A.numCoils = 1;
                end
                if size(trajectory,2) == 2 && length(size(smaps))== 3
                    A.numCoils = size(smaps, length(size(smaps)));
                end
                if size(trajectory,2) == 2 && length(size(smaps))== 2
                    A.numCoils = 1;
                end
                
                if size(trajectory,2) == 3      % 3D
                    A.smaps = reshape(smaps, [size(smaps,1), size(smaps,2), size(smaps,3), 1, size(smaps,4)]);
                else
                    A.smaps = reshape(smaps, [size(smaps,1), size(smaps,2), 1, size(smaps,3)]);
                end
                
            end
            
            if nargin<=4 || isempty(os)
                os = 1;
            end
            
            A.imageDim            = imageDim;
            A.imageDimOS          = imageDim;
            A.imageDimOS(1:end-1) = round(imageDim(1:end-1)*os);
            A.trajectory_length = size(trajectory,1);
            
            % Size of neighborhood for gridding:
            if nargin < 6 || isempty(neighbors)
                if size(trajectory,2) == 3      % 3D
                    A.nuFFTNeighbors = [5 5 5];
                else                            % 2D
                    A.nuFFTNeighbors = [5 5];
                end
            else
                A.nuFFTNeighbors = neighbors;
            end
            
            if nargin < 7 || isempty(kernel)
                kernel = 'kaiser';
            end
            
            if nargin > 7 && ~isempty(dcomp)
                A.dcomp = sqrt(dcomp(:));
            end
            
            % Siemens dimensions 2 Fessler dimensions (always fun to shuffle)
            if size(trajectory,2) == 3
                trajectory = [trajectory(:,2,:), -trajectory(:,1,:) , trajectory(:,3,:)];
            else
                trajectory = [trajectory(:,2,:), -trajectory(:,1,:)];
            end
            
            nd = size(trajectory,1);
            np = prod(A.imageDimOS(1:end-1));
            
            % Now everything is in place and we can initialize the LR_nuFFT_operator. The
            % gridding kernel can be e.g. 'kaiser' or 'minmax:kb'
            mmall = [];
            kkall = [];
            uuall = [];
            for l = 1:size(trajectory,3)
                [init, mm, kk, uu] = nufft_init(trajectory(:,:,l), A.imageDim(1:end-1), A.nuFFTNeighbors, A.imageDimOS(1:end-1), ceil(A.imageDim(1:end-1)/2), kernel);
                if l==1
                    A.sn = init.sn;
                end
                
                if nargin<3 || isempty(u) % Construct a standard nuFFT operator
                    mmall = [mmall, mm+((l-1)*nd)];
                    kkall = [kkall, kk+((l-1)*np)];
                    uuall = [uuall, uu];
                else                      % Premultiply nuFFT by u (SVD)
                    for sv = 1:size(u,2)
                        mmall = [mmall; mm+((l-1)*nd)];
                        kkall = [kkall; kk+((sv-1)*np)];
                        uuall = [uuall; u(l,sv)*uu];
                    end
                end
            end
            
            % create sparse matrix, ensuring arguments are double for stupid matlab
            A.p = sparse(mmall, kkall, uuall, nd*size(trajectory,3), prod(A.imageDimOS));
        end
        
        function s = size(A,n)
            t1 = [size(A.p,1), A.numCoils];
            t2 = A.imageDim;
            
            if A.adjoint
                tmp = t1;
                t1 = t2;
                t2 = tmp;
            end
            
            if nargin==1
                s = [prod(t1), prod(t2)];
            elseif nargin==2
                if n==1
                    s = t1;
                elseif n==2
                    s = t2;
                end
            end
        end
        
        function A = ctranspose(A)
            % Called by using A'
            A.adjoint = ~A.adjoint;
        end
        
        function Q = mtimes(A,B)
            % Called by using b = A*x; or x = A'*b;
            % The image x needs to have the dimensions [Nx Ny (Nz) R]
            % The data b needs to have the dimensions [N_samples Nt Ncoils]
            if isa(A,'LR_nuFFT_operator')
                if A.adjoint                       % This is the case A'*B
                    if ~isempty(A.dcomp)
                        B = B .* repmat(A.dcomp, [1 A.numCoils]);
                    end
                    Xk = reshape(full(A.p' * B), [A.imageDimOS A.numCoils]);
                    
                    snc = conj(A.sn);				% [*Nd,1]
                    if length(A.imageDim) == 3
                        Q = ifft(ifft(Xk,[],1),[],2);
                        Q = Q(1:A.imageDim(1),1:A.imageDim(2),:,:); % remove oversampling
                        if A.numCoils>1
                            Q = Q .* conj(A.smaps(  :,:,ones(1,A.imageDim(end)),:));
                            Q = sum(Q,4);
                        end
                        Q = Q .* snc(:,:,ones(1,A.imageDim(end))); % scaling
                    else
                        Q = ifft(ifft(ifft(Xk,[],1),[],2),[],3);
                        Q = Q(1:A.imageDim(1),1:A.imageDim(2),1:A.imageDim(3),:,:); % remove oversampling
                        if A.numCoils>1
                            Q = Q .* conj(A.smaps(:,:,:,ones(1,A.imageDim(end)),:));
                            Q = sum(Q,5);
                        end
                        Q = Q .* snc(:,:,:,ones(1,A.imageDim(end))); % scaling
                    end
                    Q = Q * sqrt(prod(A.imageDimOS(1:end-1)));
                    
                    % This is the case A*B, where B is an image that is multiplied with the
                    % coil sensitivities. Thereafter the LR_nuFFT_operator is applied
                else
                    if length(A.imageDim) == 3
                        B = B .* A.sn(:,:,ones(1,A.imageDim(end)));
                        if A.numCoils>1
                            tmp = B(  :,:,:,ones(A.numCoils,1)).*A.smaps(:,:,  ones(A.imageDim(end),1),:);
                        else
                            tmp = B;
                        end
                        tmp = reshape(fft(fft(tmp,A.imageDimOS(1),1),A.imageDimOS(2),2), [], size(tmp,4));
                    else
                        B = B .* A.sn(:,:,:,ones(1,A.imageDim(end)));
                        if A.numCoils>1
                            tmp = B(:,:,:,:,ones(A.numCoils,1)).*A.smaps(:,:,:,ones(A.imageDim(end),1),:);
                        else
                            tmp = B;
                        end
                        tmp = reshape(fft(fft(fft(tmp,A.imageDimOS(1),1),A.imageDimOS(2),2),A.imageDimOS(3),3), [], size(tmp,5));
                    end
                    Q = A.p * tmp;
                    if ~isempty(A.dcomp)
                        Q = Q .* repmat(A.dcomp, [1 A.numCoils]);
                    end
                    Q = Q / sqrt(prod(A.imageDimOS(1:end-1)));
                end
            elseif isa(B,'LR_nuFFT_operator') % now B is the operator and A is the vector
                Q = mtimes(B',A')';
            else
                error('LR_nuFFT_operator:mtimes', 'Neither A nor B is of class LR_nuFFT_operator');
            end
        end
    end
end
