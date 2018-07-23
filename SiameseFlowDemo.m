% load the images
% img1=imread('MR.bmp');
% img2=imread('CT_rotate.bmp');

% show the images
% figure;imshow(img1);title('Fixed image');
% figure;imshow(img2);title('Moving image');

% patchsize=3
% gridspacing=1

load raw.mat

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

% VISUALIZE RESULTS

% warp the image (inverse warping of img2)
[warpI2, support] = iat_pixel_warping(img2,vx,vy);

% figure;imshow(uint8(warpI2));title('Warped Image 2');

% visualize alignment error
[~, grayerror] = iat_error2gray(img1,warpI2,support);
% figure;imshow(grayerror);title('Registration error');


% display flow
siamese_flow_rgb = iat_flow2rgb(vx,vy);
% figure;imshow(iat_flow2rgb(vx,vy));title('Siamese flow field');

save source.mat