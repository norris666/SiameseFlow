load raw.mat

feature1 = single(feature1);
feature2 = single(feature2);

% SIFT-flow parameters
SIFTflowpara.alpha=2;
SIFTflowpara.d=100;
SIFTflowpara.gamma=0.005;
SIFTflowpara.nlevels=4;
SIFTflowpara.wsize=5;
SIFTflowpara.topwsize=20;
SIFTflowpara.nIterations=200;

% Run the algorithm
[vx,vy,energylist]=iat_SIFTflow(feature1,feature2,SIFTflowpara);

% warp the image (inverse warping of img2)
[warpI2, support] = iat_pixel_warping(img2,vx,vy);

% visualize alignment error
[~, grayerror] = iat_error2gray(img1,warpI2,support);

% display flow
siamese_flow_rgb = iat_flow2rgb(vx,vy);

save source.mat