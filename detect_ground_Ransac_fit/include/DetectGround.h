//
// Created by hexi on 2020/4/15.
//

#ifndef DETECT_GROUND_DETECTGROUND_H
#define DETECT_GROUND_DETECTGROUND_H

#include <iostream>
#include <opencv2/opencv.hpp>
class DetectGround{
public:
    struct BiasLine{
        float cur_k;	//当前合并的直线的K值
        float cur_b; 	//当期合并的直线的b值
        float k_sum; 	//K值的带权累加和
        float b_sum; 	//b值的带权累加和
        double weight_sum; 	//权重和
        float x1; 	//合并后大线段的左侧端点的x坐标
        float y1; 	//合并后大线段的左侧端点的y坐标
        float x2; 	//合并后大线段的右侧端点的x坐标
        float y2; 	//合并后大线段的右侧端点的y坐标
        BiasLine()
        {
            k_sum=0;
            b_sum=0;
            weight_sum=0;
        }
        bool operator >(const BiasLine &l)const
        {
            return weight_sum>l.weight_sum;
        }
    };

    struct parameter{
        int mMaxDisp;
        int mP1;
        int mP2;
        int mWinSize;
        int mPreFilterCap;
        double Gaussian_sigma;
        float mGroundLineBias;
        double min_weight;
        double max_k_distance;
        parameter()
        {
            mMaxDisp = 128;
            mP1 = 100;
            mP2 = 2700;
            mWinSize = 4;
            mPreFilterCap = 180;
            Gaussian_sigma=0.7;
            mGroundLineBias=-5;
            min_weight=20;
            max_k_distance=0.5;
        }
    };
    explicit DetectGround(cv::Mat &_left,cv::Mat &_right,cv::Mat &_disparity,parameter param);
    void compute();
    ~DetectGround();

private:
    cv::Mat left;
    cv::Mat right;
    cv::Mat mDisparity;
    cv::Mat mUDispImage;
    cv::Mat mVDispImage;
    cv::Mat mVDispBinary;
    cv::Mat mGroundmask;
    cv::Mat mGroundMapWithPlane;

    cv::Mat mDispFloat;
    parameter param;
    cv::Ptr<cv::StereoSGBM> mStereoSGBM;

    std::vector<float> mGroundLine;
    std::vector<double> mplane;
    std::vector<float> mGroundUpperBound;
    bool mGroundLineExist;
    bool isDisparity;
    void computeDisparity();
    void computeUDisparity();
    void computeVDisparity();
    bool GetGroundLine();
    void groundMaskExtraction();
    void RansacComputePlane();
    void GetSample(std::vector<double>& coeffient, std::vector<cv::Point3f> &d_list);
    void plane_fitting(std::vector<double>& coeffient, std::vector<cv::Point3f> &input,std::vector<int> &index);
    void groundPlaneRefinement();
    void  showGroundWithImage();
    void MergeBiasLines(std::vector<cv::Vec4f> &lines,std::vector<BiasLine> &cLines);
    static void calculate_line(cv::Vec4f &line,float &k, float &b);


};
#endif //DETECT_GROUND_DETECTGROUND_H
