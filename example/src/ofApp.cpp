#include "ofApp.h"

void ofApp::setup()
{
    // setup gui panel
    gui.setup("PARAMETERS");
    gui.add(mWIntensity.set("WEIGHT_INTENSITY", OFXSALIENCYMAP_DEF_WEIGHT_INTENSITY, 0, 20));
    gui.add(mWColor.set("WEIGHT_COLOT", OFXSALIENCYMAP_DEF_WEIGHT_COLOR, 0, 20));
    gui.add(mWOrientation.set("WEIGHT_ORIENTATION", OFXSALIENCYMAP_DEF_WEIGHT_ORIENTATION, 0, 20));
    gui.add(mWMotion.set("WEIGHT_MOTION", OFXSALIENCYMAP_DEF_WEIGHT_MOTION, 0, 20));
}

void ofApp::update()
{
    
}

void ofApp::draw()
{
    ofBackground(0, 0, 0);
    ofSetColor(255, 255, 255);
    
    if (mDstImg.isAllocated()) {
        
        // show image center
        ofPushMatrix();
        ofTranslate((ofGetWidth() * .5) - (mDstImg.getWidth() * .5), (ofGetHeight() * .5) - (mDstImg.getHeight() * .5));
        mDstImg.draw(0, 0);
        ofPopMatrix();
        
        stringstream s;
        s << "SPACE key: recompute saliency map" << endl;
        s << "1 key: show source image" << endl;
        s << "2 key: show saliency map image result" << endl;
        s << "3 key: show detected red image result" << endl;
        s << "4 key: show detected green image result" << endl;
        s << "5 key: show detected blue image result" << endl;
        s << "6 key: show detected intensity image result" << endl;
        s << "f key: toggle fullscreen" << endl;
        s << "l key: reload source image";
        ofSetColor(0, 255, 0);
        ofDrawBitmapString(s.str(), gui.getPosition().x + gui.getWidth(), gui.getPosition().y + 10);
        
    } else {
        
        ofDrawBitmapString("please load image ( press 'l' key )", ofGetWidth() * .5, ofGetHeight() * .5);
    }
    
    gui.draw();
}

void ofApp::keyPressed(int key)
{
    switch (key) {
        case 'l':
            loadImage();
            saliencyMap.setSourceImage(mSrcImg);    //<---------------- 1. set source image
            saliencyMap.createSaliencyMap();        //<---------------- 2. create saliency map
            mDstImg = saliencyMap.getSaliencyMap(); //<---------------- 3. get result
            break;
            
        case ' ': recompute(); break;
        case '1': mDstImg = mSrcImg; break;
        case '2': mDstImg = saliencyMap.getSaliencyMap(); break;
        case '3': mDstImg = saliencyMap.getR(); break;
        case '4': mDstImg = saliencyMap.getG(); break;
        case '5': mDstImg = saliencyMap.getB(); break;
        case '6': mDstImg = saliencyMap.getI(); break;
        case 'f': ofToggleFullscreen(); break;
    }
}

void ofApp::loadImage()
{
    ofFileDialogResult loadDialog = ofSystemLoadDialog("open",false, "../../../data");
    if (!loadDialog.bSuccess) {
        return;
    }
    if (!mSrcImg.loadImage(loadDialog.getPath())) {
        ofSystemAlertDialog("[ERROR] failed load image");
        return;
    }
    mSrcImg.setImageType(OF_IMAGE_COLOR);
}

void ofApp::recompute()
{
    if (mDstImg.isAllocated()) {
        saliencyMap.setWeightIntensity( mWIntensity );
        saliencyMap.setWeightColor( mWColor );
        saliencyMap.setWeightOrientation( mWOrientation );
        saliencyMap.setWeightMotion( mWMotion );
        
        saliencyMap.createSaliencyMap();
        
        mDstImg = saliencyMap.getSaliencyMap();
    }
}