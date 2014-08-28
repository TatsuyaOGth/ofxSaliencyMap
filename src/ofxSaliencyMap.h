/**
 ofxSaliencyMap.h https://github.com/TatsuyaOGth/ofxSaliencyMap
 
 Copyright (c) 2014 TatsuyaOGth http://ogsn.org
 
 This software is released under the MIT License.
 http://opensource.org/licenses/mit-license.php
 */
#ifndef _OFX_SALIENCY_MAP_H_
#define _OFX_SALIENCY_MAP_H_

#include "ofMain.h"
#include "ofxCv.h" //<------------------- require!

// default definition params
static const float OFXSALIENCYMAP_DEF_WEIGHT_INTENSITY      = 0.30;
static const float OFXSALIENCYMAP_DEF_WEIGHT_COLOR          = 0.30;
static const float OFXSALIENCYMAP_DEF_WEIGHT_ORIENTATION    = 0.20;
static const float OFXSALIENCYMAP_DEF_WEIGHT_MOTION         = 0.20;
static const float OFXSALIENCYMAP_DEF_RANGEMAX              = 255.00;
static const float OFXSALIENCYMAP_DEF_SCALE_GAUSS_PYRAMID   = 1.7782794100389228012254211951927;	// = 100^0.125
static const int   OFXSALIENCYMAP_DEF_DEFAULT_STEP_LOCAL    = 8;

class ofxSaliencyMap {
public:
    
    ofxSaliencyMap();
    virtual ~ofxSaliencyMap();
    
    void createSaliencyMap();
    
    void setSourceImage(const ofImage srcImg);
    void setSourceImage(const ofPixels srcPix);
    void setWeightIntensity(const float val);
    void setWeightColor(const float val);
    void setWeightOrientation(const float val);
    void setWeightMotion(const float val);

    inline ofImage getSaliencyMap(){ return mDstImg; }
    inline ofImage getR(){ return mR; }
    inline ofImage getG(){ return mG; }
    inline ofImage getB(){ return mB; }
    inline ofImage getI(){ return mI; }
    inline ofImage & getSaliencyMapRef(){ return mDstImg; }
    inline ofImage & getRRef(){ return mR; }
    inline ofImage & getGRef(){ return mG; }
    inline ofImage & getBRef(){ return mB; }
    inline ofImage & getIRef(){ return mI; }
    
private:
    
    float weightIntensity;
    float weightColor;
    float weightOrientation;
    float weightMotion;
    
    CvMat * prev_frame;
    CvMat * GaborKernel0;
    CvMat * GaborKernel45;
    CvMat * GaborKernel90;
    CvMat * GaborKernel135;
    ofImage mSrcImg;
    ofImage mDstImg;
    ofImage mR;
    ofImage mG;
    ofImage mB;
    ofImage mI;
    
    void initGabor();
    void initParams();
    
    void SMExtractRGBI(IplImage * inputImage, CvMat * &R, CvMat * &G, CvMat * &B, CvMat * &I);
    void IFMGetFM(CvMat * src, CvMat * dst[6]);
    void CFMGetFM(CvMat * R, CvMat * G, CvMat * B, CvMat * RGFM[6], CvMat * BYFM[6]);
    void OFMGetFM(CvMat * I, CvMat * dst[24]);
    void MFMGetFM(CvMat * I, CvMat * dst_x[6], CvMat * dst_y[6]);
    void normalizeFeatureMaps(CvMat * FM[6], CvMat * NFM[6], int width, int height, int num_maps);
    CvMat * SMNormalization(CvMat * src);	// Itti normalization
    CvMat * SMRangeNormalize(CvMat * src);	// dynamic range normalization
    CvMat * ICMGetCM(CvMat *IFM[6], CvSize size);
    CvMat * CCMGetCM(CvMat *CFM_RG[6], CvMat *CFM_BY[6], CvSize size);
    CvMat * OCMGetCM(CvMat *OFM[24], CvSize size);
    CvMat * MCMGetCM(CvMat *MFM_X[6], CvMat *MFM_Y[6], CvSize size);
    
};
#endif
