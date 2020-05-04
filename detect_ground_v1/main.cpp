#include <iostream>
#include <opencv2/opencv.hpp>
#include <DetectGround.h>
using namespace std;

int main() {
    string left_base_dir="../image_00/data/";
    string right_base_dir="../image_01/data/";
    string disp_base_dir="../disp/disp_";
    for(int i=0;i<153;i++)
    {
        cv::Mat left=cv::imread(left_base_dir+cv::format("%010d.png",i));
        cv::Mat right=cv::imread(right_base_dir+cv::format("%010d.png",i));
        cv::Mat disp;
        cv::Mat disp64=cv::imread(disp_base_dir+cv::format("%010d.png",i),cv::IMREAD_UNCHANGED);
        disp64.convertTo(disp, CV_8UC1, 1.0 / 256);
        DetectGround::parameter param;
        DetectGround ground(left,right,disp,param);
        ground.compute();
        cv::waitKey(10);
    }

    return 0;
}