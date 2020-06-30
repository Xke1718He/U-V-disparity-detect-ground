//
// Created by hexi on 2020/4/15.
//

#ifndef DETECT_GROUND_DETECTGROUND_H
#define DETECT_GROUND_DETECTGROUND_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <DBSCAN.hpp>
#include<cmath>
class DetectGround{
public:
    struct parameter{
        int mMaxDisp;
        int mP1;
        int mP2;
        int mWinSize;
        int mPreFilterCap;
        parameter()
        {
            mMaxDisp = 128;
            mP1 = 100;
            mP2 = 2700;
            mWinSize = 4;
            mPreFilterCap = 180;
        }
    };
    explicit DetectGround(parameter param,cv::Mat &left,cv::Mat &right);
    void computeDisparity();
    void computeUDisparity(float b);
    void computeVDisparity();
    void DpOptimize(cv::Mat sum ,cv::Mat dir,cv::Mat Udisparity);
    void Detect();

    ~DetectGround();

private:
    cv::Mat left;
    cv::Mat right;
    cv::Mat disp;
    cv::Mat mUDispImage;
    cv::Mat mVDispImage;
    cv::Mat mVDispBinary;
    cv::Mat mUDispBinary;
    cv::Mat mDispWithoutGround;

    parameter param;
    cv::Ptr<cv::StereoSGBM> mStereoSGBM;
    std::vector<cv::Point> points;

};
#endif //DETECT_GROUND_DETECTGROUND_H
