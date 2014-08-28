/**
 ofxSaliencyMap.cpp https://github.com/TatsuyaOGth/ofxSaliencyMap
 
 Copyright (c) 2014 TatsuyaOGth http://ogsn.org
 
 This software is released under the MIT License.
 http://opensource.org/licenses/mit-license.php
 */
#include "ofxSaliencyMap.h"

using namespace ofxCv;
using namespace cv;

void FMGaussianPyrCSD(CvMat* src, CvMat* dst[6]);
void FMCreateGaussianPyr(CvMat* src, CvMat* dst[9]);
void FMCenterSurroundDiff(CvMat* GaussianMap[9], CvMat* dst[6]);
double SMAvgLocalMax(CvMat* src);

ofxSaliencyMap::ofxSaliencyMap()
{
    prev_frame = 0;
    GaborKernel0 = 0;
    GaborKernel45 = 0;
    GaborKernel90 = 0;
    GaborKernel135 = 0;
    initParams();
}

ofxSaliencyMap::~ofxSaliencyMap()
{
    cvReleaseMat(&prev_frame);
    cvReleaseMat(&GaborKernel0);
    cvReleaseMat(&GaborKernel45);
    cvReleaseMat(&GaborKernel90);
    cvReleaseMat(&GaborKernel135);
}

void ofxSaliencyMap::createSaliencyMap()
{
    // check source image
    if (!mSrcImg.isAllocated()) {
        cout << "[ERROR] do not read source image" << endl;
        return;
    }
    
    IplImage src = toCv(mSrcImg);
    
    // init gabor kernels
    initGabor();
    
    CvSize sSize = cvSize(mSrcImg.width, mSrcImg.height);

    //----------
    // Intensity and RGB Extraction
    //----------
    
    CvMat * R, * G, * B, * I;
    SMExtractRGBI(&src, R, G, B, I);
    
    // intensity feature maps
    CvMat* IFM[6];
    IFMGetFM(I, IFM);
    
    // color feature maps
    CvMat* CFM_RG[6];
    CvMat* CFM_BY[6];
    CFMGetFM(R, G, B, CFM_RG, CFM_BY);
    
    // orientation feature maps
    CvMat* OFM[24];
    OFMGetFM(I, OFM);
    
    // motion feature maps
    CvMat* MFM_X[6];
    CvMat* MFM_Y[6];
    MFMGetFM(I, MFM_X, MFM_Y);
    
    // output RGB and I images
    CvMat *tmpR = cvCreateMat(sSize.height, sSize.width, CV_8UC1);
    CvMat *tmpG = cvCreateMat(sSize.height, sSize.width, CV_8UC1);
    CvMat *tmpB = cvCreateMat(sSize.height, sSize.width, CV_8UC1);
    CvMat *tmpI = cvCreateMat(sSize.height, sSize.width, CV_8UC1);
    cvConvertScaleAbs(R, tmpR, 255);
    cvConvertScaleAbs(G, tmpG, 255);
    cvConvertScaleAbs(B, tmpB, 255);
    cvConvertScaleAbs(I, tmpI, 255);
    mR.setFromPixels((unsigned char*)tmpR->data.ptr, sSize.width, sSize.height, OF_IMAGE_GRAYSCALE);
    mG.setFromPixels((unsigned char*)tmpG->data.ptr, sSize.width, sSize.height, OF_IMAGE_GRAYSCALE);
    mB.setFromPixels((unsigned char*)tmpB->data.ptr, sSize.width, sSize.height, OF_IMAGE_GRAYSCALE);
    mI.setFromPixels((unsigned char*)tmpI->data.ptr, sSize.width, sSize.height, OF_IMAGE_GRAYSCALE);
    
    cvReleaseMat(&tmpR);
    cvReleaseMat(&tmpG);
    cvReleaseMat(&tmpB);
    cvReleaseMat(&tmpI);
    cvReleaseMat(&R);
    cvReleaseMat(&G);
    cvReleaseMat(&B);
    cvReleaseMat(&I);
    
    //----------
    // Generate Conspicuity Map
    //----------

    CvMat *ICM = ICMGetCM(IFM, sSize);
    CvMat *CCM = CCMGetCM(CFM_RG, CFM_BY, sSize);
    CvMat *OCM = OCMGetCM(OFM, sSize);
    CvMat *MCM = MCMGetCM(MFM_X, MFM_Y, sSize);
    
    for(int i=0; i<6; i++){
        
        cvReleaseMat(&IFM[i]);
        cvReleaseMat(&CFM_RG[i]);
        cvReleaseMat(&CFM_BY[i]);
        cvReleaseMat(&MFM_X[i]);
        cvReleaseMat(&MFM_Y[i]);
        
    }
    for(int i=0; i<24; i++) cvReleaseMat(&OFM[i]);
    
    //----------
    // Generate Saliency Map
    //----------
    
    // Normalize conspicuity maps
    CvMat *ICM_norm;
    CvMat *CCM_norm;
    CvMat *OCM_norm;
    CvMat *MCM_norm;
    ICM_norm = SMNormalization(ICM);
    CCM_norm = SMNormalization(CCM);
    OCM_norm = SMNormalization(OCM);
    MCM_norm = SMNormalization(MCM);
    cvReleaseMat(&ICM);
    cvReleaseMat(&CCM);
    cvReleaseMat(&OCM);
    cvReleaseMat(&MCM);
    
    // Adding all the CMs to form Saliency Map
    CvMat* SM_Mat = cvCreateMat(sSize.height, sSize.width, CV_32FC1);
    cvAddWeighted(ICM_norm, weightIntensity, OCM_norm, weightOrientation, 0.0, SM_Mat);
    cvAddWeighted(CCM_norm, weightColor, SM_Mat, 1.00, 0.0, SM_Mat);
    cvAddWeighted(MCM_norm, weightMotion, SM_Mat, 1.00, 0.0, SM_Mat);
    cvReleaseMat(&ICM_norm);
    cvReleaseMat(&CCM_norm);
    cvReleaseMat(&OCM_norm);
    cvReleaseMat(&MCM_norm);
    
    // Output Result Map
    CvMat *cvtMat = cvCreateMat(sSize.height, sSize.width, CV_8UC1);
    CvMat *SM = SMRangeNormalize(SM_Mat);
    cvConvertScaleAbs(SM, cvtMat, 255);
    mDstImg.setFromPixels((unsigned char *)cvtMat->data.ptr, cvtMat->cols, cvtMat->rows, OF_IMAGE_GRAYSCALE);
    
    cvReleaseMat(&SM_Mat);
    cvReleaseMat(&SM);
    cvReleaseMat(&cvtMat);
    
}

void ofxSaliencyMap::SMExtractRGBI(IplImage* inputImage, CvMat* &R, CvMat* &G, CvMat* &B, CvMat* &I)
{
    
    int height = inputImage->height;
    int width = inputImage->width;
    // convert scale of array elements
    CvMat * src = cvCreateMat(height, width, CV_32FC3);
    cvConvertScale(inputImage, src, 1/255.0);
    
    // initalize matrix for I,R,G,B
    R = cvCreateMat(height, width, CV_32FC1);
    G = cvCreateMat(height, width, CV_32FC1);
    B = cvCreateMat(height, width, CV_32FC1);
    I = cvCreateMat(height, width, CV_32FC1);
    
    // split
    cvSplit(src, B, G, R, NULL);
    
    // extract intensity image
    cvCvtColor(src, I, CV_BGR2GRAY);
    
    // release
    cvReleaseMat(&src);
    
}

void ofxSaliencyMap::IFMGetFM(CvMat* src, CvMat* dst[6])
{
    
    FMGaussianPyrCSD(src, dst);
    
}

void ofxSaliencyMap::CFMGetFM(CvMat* R, CvMat* G, CvMat* B, CvMat* RGFM[6], CvMat* BYFM[6])
{
    
    // allocate
    int height = R->height;
    int width = R->width;
    CvMat* tmp1 = cvCreateMat(height, width, CV_32FC1);
    CvMat* tmp2 = cvCreateMat(height, width, CV_32FC1);
    CvMat* RGBMax = cvCreateMat(height, width, CV_32FC1);
    CvMat* RGMin = cvCreateMat(height, width, CV_32FC1);
    CvMat* RGMat = cvCreateMat(height, width, CV_32FC1);
    CvMat* BYMat = cvCreateMat(height, width, CV_32FC1);
    // Max(R,G,B)
    cvMax(R, G, tmp1);
    cvMax(B, tmp1, RGBMax);
    cvMaxS(RGBMax, 0.0001, RGBMax); // to prevent dividing by 0
    // Min(R,G)
    cvMin(R, G, RGMin);
    
    // R-G
    cvSub(R, G, tmp1);
    // B-Min(R,G)
    cvSub(B, RGMin, tmp2);
    // RG = (R-G)/Max(R,G,B)
    cvDiv(tmp1, RGBMax, RGMat);
    // BY = (B-Min(R,G)/Max(R,G,B)
    cvDiv(tmp2, RGBMax, BYMat);
    
    // Clamp negative value to 0 for the RG and BY maps
    cvMaxS(RGMat, 0, RGMat);
    cvMaxS(BYMat, 0, BYMat);
    
    // Obtain [RG,BY] color opponency feature map by generating Gaussian pyramid and performing center-surround difference
    FMGaussianPyrCSD(RGMat, RGFM);
    FMGaussianPyrCSD(BYMat, BYFM);
    
    // release
    cvReleaseMat(&tmp1);
    cvReleaseMat(&tmp2);
    cvReleaseMat(&RGBMax);
    cvReleaseMat(&RGMin);
    cvReleaseMat(&RGMat);
    cvReleaseMat(&BYMat);
    
}

void ofxSaliencyMap::OFMGetFM(CvMat* I, CvMat* dst[24])
{
    
    // Create gaussian pyramid
    CvMat* GaussianI[9];
    FMCreateGaussianPyr(I, GaussianI);
    
    // Convolution Gabor filter with intensity feature maps to extract orientation feature
    CvMat* tempGaborOutput0[9];
    CvMat* tempGaborOutput45[9];
    CvMat* tempGaborOutput90[9];
    CvMat* tempGaborOutput135[9];
    for(int j=2; j<9; j++)
    {
        
        int now_height = GaussianI[j]->height;
        int now_width = GaussianI[j]->width;
        tempGaborOutput0[j] = cvCreateMat(now_height, now_width, CV_32FC1);
        tempGaborOutput45[j] = cvCreateMat(now_height, now_width, CV_32FC1);
        tempGaborOutput90[j] = cvCreateMat(now_height, now_width, CV_32FC1);
        tempGaborOutput135[j] = cvCreateMat(now_height, now_width, CV_32FC1);
        cvFilter2D(GaussianI[j], tempGaborOutput0[j], GaborKernel0);
        cvFilter2D(GaussianI[j], tempGaborOutput45[j], GaborKernel45);
        cvFilter2D(GaussianI[j], tempGaborOutput90[j], GaborKernel90);
        cvFilter2D(GaussianI[j], tempGaborOutput135[j], GaborKernel135);
        
    }
    for(int j=0; j<9; j++) cvReleaseMat(&(GaussianI[j]));
    
    // calculate center surround difference for each orientation
    CvMat* temp0[6];
    CvMat* temp45[6];
    CvMat* temp90[6];
    CvMat* temp135[6];
    FMCenterSurroundDiff(tempGaborOutput0, temp0);
    FMCenterSurroundDiff(tempGaborOutput45, temp45);
    FMCenterSurroundDiff(tempGaborOutput90, temp90);
    FMCenterSurroundDiff(tempGaborOutput135, temp135);
    for(int i=2; i<9; i++)
    {
        
        cvReleaseMat(&(tempGaborOutput0[i]));
        cvReleaseMat(&(tempGaborOutput45[i]));
        cvReleaseMat(&(tempGaborOutput90[i]));
        cvReleaseMat(&(tempGaborOutput135[i]));
        
    }
    // saving the 6 center-surround difference feature map of each angle configuration to the destination pointer
    for(int i=0; i<6; i++)
    {
        
        dst[i] = temp0[i];
        dst[i+6] = temp45[i];
        dst[i+12] = temp90[i];
        dst[i+18] = temp135[i];
        
    }
    
}

void ofxSaliencyMap::MFMGetFM(CvMat* I, CvMat* dst_x[], CvMat* dst_y[])
{
    
    int height = I->height;
    int width = I->width;
    // convert
    CvMat* I8U = cvCreateMat(height, width, CV_8UC1);
    cvConvertScale(I, I8U, 256);
    
    // obtain optical flow information
    CvMat* flowx = cvCreateMat(height, width, CV_32FC1);
    CvMat* flowy = cvCreateMat(height, width, CV_32FC1);
    cvSetZero(flowx);
    cvSetZero(flowy);
    if(this->prev_frame!=NULL)
    {
        
        cvCalcOpticalFlowLK(this->prev_frame, I8U, cvSize(7,7), flowx, flowy);
        cvReleaseMat(&(this->prev_frame));
        
    }
    // create Gaussian pyramid
    FMGaussianPyrCSD(flowx, dst_x);
    FMGaussianPyrCSD(flowy, dst_y);
    
    // update
    this->prev_frame = cvCloneMat(I8U);
    
    // release
    cvReleaseMat(&flowx);
    cvReleaseMat(&flowy);
    cvReleaseMat(&I8U);
    
}

void FMGaussianPyrCSD(CvMat* src, CvMat* dst[6])
{
    
    CvMat *GaussianMap[9];
    FMCreateGaussianPyr(src, GaussianMap);
    FMCenterSurroundDiff(GaussianMap, dst);
    for(int i=0; i<9; i++) cvReleaseMat(&(GaussianMap[i]));
    
}

void FMCreateGaussianPyr(CvMat* src, CvMat* dst[9])
{
    
    dst[0] = cvCloneMat(src);
    for(int i=1; i<9; i++)
    {
        
        dst[i] = cvCreateMat(dst[i-1]->height/2, dst[i-1]->width/2, CV_32FC1);
        cvPyrDown(dst[i-1], dst[i], CV_GAUSSIAN_5x5);
        
    }
    
}

void FMCenterSurroundDiff(CvMat* GaussianMap[9], CvMat* dst[6])
{
    
    int i=0;
    for(int s=2; s<5; s++)
    {
        
        int now_height  = GaussianMap[s]->height;
        int now_width   = GaussianMap[s]->width;
        CvMat * tmp     = cvCreateMat(now_height, now_width, CV_32FC1);
        dst[i]          = cvCreateMat(now_height, now_width, CV_32FC1);
        dst[i+1]        = cvCreateMat(now_height, now_width, CV_32FC1);
        cvResize(GaussianMap[s+3], tmp, CV_INTER_LINEAR);
        cvAbsDiff(GaussianMap[s], tmp, dst[i]);
        cvResize(GaussianMap[s+4], tmp, CV_INTER_LINEAR);
        cvAbsDiff(GaussianMap[s], tmp, dst[i+1]);
        cvReleaseMat(&tmp);
        i += 2;
        
    }
    
}

void ofxSaliencyMap::normalizeFeatureMaps(CvMat *FM[], CvMat *NFM[], int width, int height, int num_maps)
{
    
    for(int i=0; i<num_maps; i++)
    {
        
        CvMat * normalizedImage = SMNormalization(FM[i]);
        NFM[i] = cvCreateMat(height, width, CV_32FC1);
        cvResize(normalizedImage, NFM[i], CV_INTER_LINEAR);
        cvReleaseMat(&normalizedImage);
        
    }
    
}
CvMat* ofxSaliencyMap::SMNormalization(CvMat* src)
{
    
    CvMat* result = cvCreateMat(src->height, src->width, CV_32FC1);
    
    // normalize so that the pixel value lies between 0 and 1
    CvMat* tempResult = SMRangeNormalize(src);
    // single-peak emphasis / multi-peak suppression
    double lmaxmean = SMAvgLocalMax(tempResult);
    double normCoeff = (1-lmaxmean)*(1-lmaxmean);
    cvConvertScale(tempResult, result, normCoeff);
    cvReleaseMat(&tempResult);
    return result;
    
}
CvMat* ofxSaliencyMap::SMRangeNormalize(CvMat* src)
{
    
    double maxx, minn;
    cvMinMaxLoc(src, &minn, &maxx);
    CvMat* result = cvCreateMat(src->height, src->width, CV_32FC1);
    if(maxx!=minn) cvConvertScale(src, result, 1/(maxx-minn), minn/(minn-maxx));
    else cvConvertScale(src, result, 1, -minn);
    return result;
    
}
double SMAvgLocalMax(CvMat* src)
{
    
    int stepsize = OFXSALIENCYMAP_DEF_DEFAULT_STEP_LOCAL;
    int numlocal = 0;
    double lmaxmean = 0, lmax = 0, dummy = 0;
    CvMat localMatHeader;
    cvInitMatHeader(&localMatHeader, stepsize, stepsize, CV_32FC1, src->data.ptr, src->step);
    for(int y=0; y<src->height-stepsize; y+=stepsize) // Note: the last several pixels may be ignored.
    {
        
        for(int x=0; x<src->width-stepsize; x+=stepsize)
        {
            
            localMatHeader.data.ptr = src->data.ptr+sizeof(float)*x+src->step*y;	// get local matrix by pointer trick
            cvMinMaxLoc(&localMatHeader, &dummy, &lmax);
            lmaxmean += lmax;
            numlocal++;
            
        }
        
    }
    return lmaxmean/numlocal;
    
}

CvMat * ofxSaliencyMap::ICMGetCM(CvMat *IFM[], CvSize size)
{
    
    int num_FMs = 6;
    // Normalize all intensity feature maps
    CvMat * NIFM[6];
    normalizeFeatureMaps(IFM, NIFM, size.width, size.height, num_FMs);
    
    // Formulate intensity conspicuity map by summing up the normalized intensity feature maps
    CvMat *ICM = cvCreateMat(size.height, size.width, CV_32FC1);
    cvSetZero(ICM);
    for (int i=0; i<num_FMs; i++)
    {
        
        cvAdd(ICM, NIFM[i], ICM);
        cvReleaseMat(&NIFM[i]);
        
    }
    return ICM;
    
}
CvMat * ofxSaliencyMap::CCMGetCM(CvMat *CFM_RG[], CvMat *CFM_BY[], CvSize size)
{
    
//    int num_FMs = 6;
    CvMat* CCM_RG = ICMGetCM(CFM_RG, size);
    CvMat* CCM_BY = ICMGetCM(CFM_BY, size);
    CvMat *CCM = cvCreateMat(size.height, size.width, CV_32FC1);
    cvAdd(CCM_BY, CCM_RG, CCM);
    
    cvReleaseMat(&CCM_BY);
    cvReleaseMat(&CCM_RG);
    
    return CCM;
    
}
CvMat * ofxSaliencyMap::OCMGetCM(CvMat *OFM[], CvSize size)
{
    
    int num_FMs_perAngle = 6;
//    int num_angles = 4;
//    int num_FMs = num_FMs_perAngle * num_angles;
    // split feature maps into four sets
    CvMat * OFM0[6];
    CvMat * OFM45[6];
    CvMat * OFM90[6];
    CvMat * OFM135[6];
    for (int i=0; i<num_FMs_perAngle; i++)
    {
        
        OFM0[i] = OFM[0*num_FMs_perAngle+i];
        OFM45[i] = OFM[1*num_FMs_perAngle+i];
        OFM90[i] = OFM[2*num_FMs_perAngle+i];
        OFM135[i] = OFM[3*num_FMs_perAngle+i];
        
    }
    // extract conspicuity map for each angle
    CvMat * NOFM_tmp[4];
    NOFM_tmp[0] = ICMGetCM(OFM0, size);
    NOFM_tmp[1] = ICMGetCM(OFM45, size);
    NOFM_tmp[2] = ICMGetCM(OFM90, size);
    NOFM_tmp[3] = ICMGetCM(OFM135, size);
    
    // Normalize all orientation features map grouped by their orientation angles
    CvMat* NOFM[4];
    for (int i=0; i<4; i++)
    {
        
        NOFM[i] = SMNormalization(NOFM_tmp[i]);
        cvReleaseMat(&NOFM_tmp[i]);
        
    }
    // Sum up all orientation feature maps, and form orientation conspicuity map
    CvMat *OCM = cvCreateMat(size.height, size.width, CV_32FC1);
    cvSetZero(OCM);
    for(int i=0; i<4; i++)
    {
        
        cvAdd(NOFM[i], OCM, OCM);
        cvReleaseMat(&NOFM[i]);
        
    }
    return OCM;
    
}
CvMat * ofxSaliencyMap::MCMGetCM(CvMat *MFM_X[], CvMat *MFM_Y[], CvSize size)
{
    return CCMGetCM(MFM_X, MFM_Y, size);
}

void ofxSaliencyMap::initGabor()
{
    static const double	GaborKernel_0[9][9] = {
        
        {1.85212E-06,	1.28181E-05,	-0.000350433,	-0.000136537, 0.002010422,	-0.000136537,	-0.000350433,	1.28181E-05, 1.85212E-06},
        {2.80209E-05,	0.000193926,	-0.005301717,	-0.002065674, 0.030415784,	-0.002065674,	-0.005301717,	0.000193926, 2.80209E-05},
        {0.000195076,	0.001350077,	-0.036909595,   -0.014380852,   0.211749204,	-0.014380852,	-0.036909595, 0.001350077,	0.000195076},
        {0.00062494,	0.004325061,	-0.118242318,	-0.046070008, 0.678352526,	-0.046070008,	-0.118242318,	0.004325061, 0.00062494},
        {0.000921261,	0.006375831,	-0.174308068, -0.067914552,	1,	 -0.067914552,	-0.174308068, 0.006375831,	0.000921261},
        {0.00062494,	0.004325061,	-0.118242318,	-0.046070008, 0.678352526,	-0.046070008,	-0.118242318,	0.004325061, 0.00062494},
        {0.000195076,	0.001350077,	-0.036909595, -0.014380852,	0.211749204,	-0.014380852,	-0.036909595, 0.001350077,	0.000195076},
        {2.80209E-05,	0.000193926,	-0.005301717,	-0.002065674, 0.030415784,	-0.002065674,	-0.005301717,	0.000193926, 2.80209E-05},
        {1.85212E-06,	1.28181E-05,	-0.000350433,	-0.000136537, 0.002010422,	-0.000136537,	-0.000350433,	1.28181E-05, 1.85212E-06}
        
    };
    static const double	GaborKernel_45[9][9] = {
        
        {4.0418E-06,	2.2532E-05,	 -0.000279806,	-0.001028923, 3.79931E-05,	0.000744712,	0.000132863,	-9.04408E-06, -1.01551E-06},
        {2.2532E-05,	0.00092512,	 0.002373205,	-0.013561362, -0.0229477,	 0.000389916,	0.003516954	,	0.000288732, -9.04408E-06},
        {-0.000279806,	0.002373205,	0.044837725,	0.052928748, -0.139178011,	-0.108372072,	0.000847346	,	0.003516954, 0.000132863},
        {-0.001028923,	-0.013561362,	0.052928748,	0.46016215, 0.249959607,	-0.302454279,	-0.108372072,	0.000389916, 0.000744712},
        {3.79931E-05,	-0.0229477,	 -0.139178011,	0.249959607, 1,	 0.249959607,	-0.139178011,	-0.0229477,	 3.79931E-05},
        {0.000744712,	0.000389916,	-0.108372072, -0.302454279,	0.249959607,	0.46016215,	 0.052928748, -0.013561362,	-0.001028923},
        {0.000132863,	0.003516954,	0.000847346,	-0.108372072, -0.139178011,	0.052928748,	0.044837725,	0.002373205, -0.000279806},
        {-9.04408E-06,	0.000288732,	0.003516954,	0.000389916, -0.0229477,	 -0.013561362,	0.002373205,	0.00092512, 2.2532E-05},
        {-1.01551E-06,	-9.04408E-06,	0.000132863,	0.000744712, 3.79931E-05,	-0.001028923,	-0.000279806,	2.2532E-05, 4.0418E-06}
        
    };
    static const double GaborKernel_90[9][9] = {
        
        {1.85212E-06,	2.80209E-05,	0.000195076,	0.00062494, 0.000921261,	0.00062494,	 0.000195076,	2.80209E-05, 1.85212E-06},
        {1.28181E-05,	0.000193926,	0.001350077,	0.004325061, 0.006375831,	0.004325061,	0.001350077,	0.000193926, 1.28181E-05},
        {-0.000350433,	-0.005301717,	-0.036909595, -0.118242318,	-0.174308068,	-0.118242318, -0.036909595,	-0.005301717,	-0.000350433},
        {-0.000136537,	-0.002065674,	-0.014380852, -0.046070008,	-0.067914552,	-0.046070008, -0.014380852,	-0.002065674,	-0.000136537},
        {0.002010422,	0.030415784,	0.211749204,	0.678352526, 1,	 0.678352526,	0.211749204,	0.030415784, 0.002010422},
        {-0.000136537,	-0.002065674,	-0.014380852, -0.046070008,	-0.067914552,	-0.046070008, -0.014380852,	-0.002065674,	-0.000136537},
        {-0.000350433,	-0.005301717,	-0.036909595, -0.118242318,	-0.174308068,	-0.118242318, -0.036909595,	-0.005301717,	-0.000350433},
        {1.28181E-05,	0.000193926,	0.001350077,	0.004325061, 0.006375831,	0.004325061,	0.001350077,	0.000193926, 1.28181E-05},
        {1.85212E-06,	2.80209E-05,	0.000195076,	0.00062494, 0.000921261,	0.00062494,	 0.000195076,	2.80209E-05, 1.85212E-06}
        
    };
    static const double	GaborKernel_135[9][9] = {
        
        {-1.01551E-06,	-9.04408E-06,	0.000132863,	0.000744712, 3.79931E-05,	-0.001028923,	-0.000279806,	2.2532E-05, 4.0418E-06},
        {-9.04408E-06,	0.000288732,	0.003516954,	0.000389916, -0.0229477,	 -0.013561362,	0.002373205,	0.00092512, 2.2532E-05},
        {0.000132863,	0.003516954,	0.000847346,	-0.108372072, -0.139178011,	0.052928748,	0.044837725,	0.002373205, -0.000279806},
        {0.000744712,	0.000389916,	-0.108372072, -0.302454279,	0.249959607,	0.46016215,	 0.052928748, -0.013561362,	-0.001028923},
        {3.79931E-05,	-0.0229477,	 -0.139178011,	0.249959607, 1,	 0.249959607,	-0.139178011,	-0.0229477,	 3.79931E-05},
        {-0.001028923,	-0.013561362,	0.052928748,	0.46016215, 0.249959607	,	-0.302454279,	-0.108372072,	0.000389916, 0.000744712},
        {-0.000279806,	0.002373205,	0.044837725,	0.052928748, -0.139178011,	-0.108372072,	0.000847346,	0.003516954, 0.000132863},
        {2.2532E-05,	0.00092512,	 0.002373205,	-0.013561362, -0.0229477,	 0.000389916,	0.003516954,	0.000288732, -9.04408E-06},
        {4.0418E-06,	2.2532E-05,	 -0.000279806,	-0.001028923, 3.79931E-05	,	0.000744712,	0.000132863,	-9.04408E-06, -1.01551E-06}
        
    };
    
    // previous frame information
    prev_frame = NULL;
    //Set Gabor Kernel (9x9)
    GaborKernel0 = cvCreateMat(9, 9, CV_32FC1);
    GaborKernel45 = cvCreateMat(9, 9, CV_32FC1);
    GaborKernel90 = cvCreateMat(9, 9, CV_32FC1);
    GaborKernel135 = cvCreateMat(9, 9, CV_32FC1);
    for(int i=0; i<9; i++) for(int j=0; j<9; j++){
        cvmSet(GaborKernel0, i, j, GaborKernel_0[i][j]);	 // 0 degree orientation
        cvmSet(GaborKernel45, i, j, GaborKernel_45[i][j]); // 45 degree orientation
        cvmSet(GaborKernel90, i, j, GaborKernel_90[i][j]); // 90 degree orientation
        cvmSet(GaborKernel135, i, j, GaborKernel_135[i][j]); // 135 degree orientation
    }
}

void ofxSaliencyMap::initParams()
{
    
    setWeightIntensity(OFXSALIENCYMAP_DEF_WEIGHT_INTENSITY);
    setWeightColor(OFXSALIENCYMAP_DEF_WEIGHT_COLOR);
    setWeightOrientation(OFXSALIENCYMAP_DEF_WEIGHT_ORIENTATION);
    setWeightMotion(OFXSALIENCYMAP_DEF_WEIGHT_MOTION);

}

//////////////////////////////////////////////////////////////////
// Getter and Setter
//////////////////////////////////////////////////////////////////
void ofxSaliencyMap::setSourceImage(ofImage srcImg)
{
    if (srcImg.isAllocated()) {
        mSrcImg = srcImg;
    }
}

void ofxSaliencyMap::setSourceImage(ofPixels srcPix)
{
    if (srcPix.isAllocated()) {
        mSrcImg.setFromPixels(srcPix);
    }
}

void ofxSaliencyMap::setWeightIntensity(const float val)
{
    weightIntensity = val;
}

void ofxSaliencyMap::setWeightColor(const float val)
{
    weightColor = val;
}

void ofxSaliencyMap::setWeightOrientation(const float val)
{
    weightOrientation = val;
}

void ofxSaliencyMap::setWeightMotion(const float val)
{
    weightMotion = val;
}
