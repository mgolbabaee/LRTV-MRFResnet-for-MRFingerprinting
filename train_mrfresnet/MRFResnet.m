function lgraph = MRFResnet(datach, nbch1, nbch2, depth_out, depth_int)
%
% The MRFResnet encoder network proposed in 
%
%     M. Golbabaee, G. Bounincontri, C. Pirkl, M. Menzel, B. Menze, 
%     M. Davies, and P. Gomez. "Compressive MRI quantification using convex
%     spatiotemporal priors and deep auto-encoders." arXiv preprint arXiv:2001.08746 (2020).
%
% (c) 2018-2020 Mohammad Golbabaee, m.golbabaee@bath.ac.uk
%% 
layers = [
    imageInputLayer([1 1 datach],'Name','relu_in2') 
        ];


for i=1:depth_out
    layers = [layers
        RSB(nbch1, nbch2, depth_int ,['RSB',num2str(i+1)])
        reluLayer('Name',['relu',num2str(i+1)])];
end

layers = [layers
    convolution2dLayer(1, 2,'Padding',0,'Name','conv_out')
    reluLayer('Name','sigm_out')
    regressionLayer('Name','reg_out')
    ];

lgraph = layerGraph(layers);

lgraph = connectLayers(lgraph,'relu_in2','RSB2_add/in2');

for i=2:depth_out
    lgraph = connectLayers(lgraph,['relu',num2str(i)],['RSB',num2str(i+1),'_add/in2']);
end
plot(lgraph)

end