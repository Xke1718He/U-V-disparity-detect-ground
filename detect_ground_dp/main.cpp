#include <iostream>
#include <opencv2/opencv.hpp>
#include <DetectGround.h>
using namespace std;

int main() {
    //SGBM param
    DetectGround::parameter param;
    string left_base_dir="../2009_09_08_drive_0010_Images/I1_";
    string right_base_dir="../2009_09_08_drive_0010_Images/I2_";
    string disp_base_dir="../disp/disp_";
    for(int i=0;i<1423;i++) {
        cv::Mat left = cv::imread(left_base_dir + cv::format("%06d.png", i));
        cv::Mat right = cv::imread(right_base_dir + cv::format("%06d.png", i));
        DetectGround ground(param, left, right);
        ground.computeDisparity();
        ground.computeVDisparity();
        ground.Detect();
        cv::waitKey(10);
    }
    return 0;
}