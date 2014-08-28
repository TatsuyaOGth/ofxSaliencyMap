#pragma once

#include "ofMain.h"
#include "ofxSaliencyMap.h"
#include "ofxGui.h"

class ofApp : public ofBaseApp{
    
public:
    void setup();
    void update();
    void draw();
    void keyPressed(int key);
    
    void loadImage();
    void recompute();
    
    
    ofxSaliencyMap saliencyMap;
    
    ofImage mSrcImg;
    ofImage mDstImg;
    
    ofParameter<float> mWIntensity;
    ofParameter<float> mWColor;
    ofParameter<float> mWOrientation;
    ofParameter<float> mWMotion;
    
    ofxPanel gui;
};
