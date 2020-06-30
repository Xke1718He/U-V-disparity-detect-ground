//
// Created by hexi on 2020/4/15.
//
#include <DetectGround.h>
#include <opencv2/line_descriptor/descriptor.hpp>
#include "opencv2/ximgproc.hpp"
#include <cmath>

DetectGround::DetectGround(cv::Mat &_left, cv::Mat &_right,cv::Mat &_disparity,DetectGround::parameter param) {
    left=_left;
    right=_right;
    mDisparity=_disparity;
    this->param=param;
    mStereoSGBM=cv::StereoSGBM::create(0,param.mMaxDisp,param.mWinSize,param.mP1,param.mP2,0,param.mPreFilterCap,5,0,0,cv::StereoSGBM::MODE_HH);
    if(!mDisparity.empty())
    {
        double min,max;
        cv::minMaxLoc(mDisparity, &min, &max);
        param.mMaxDisp=(int)max;
        isDisparity=true;
    } else
    {
        isDisparity= false;
    }
}
DetectGround::~DetectGround() {

}
void DetectGround::compute() {
    if(!isDisparity)
        computeDisparity();
    computeVDisparity();
    computeUDisparity();
    GetGroundLine();
    groundMaskExtraction();
    RansacComputePlane();
    groundPlaneRefinement();
    showGroundWithImage();
}
//计算视差图像
void DetectGround::computeDisparity() {
    //compute disparity
    cv::Mat disp16s;
    mStereoSGBM->compute(left,right,disp16s);
    disp16s=disp16s/16;
    disp16s.convertTo(mDisparity,CV_8UC1);
    //DEBUG
//    cv::imshow("disparity",mDisparity);
}

//compute U disparity image
void DetectGround::computeUDisparity() {
    //init u disparity image
    if(mUDispImage.empty())
        mUDispImage.create(param.mMaxDisp, mDisparity.cols, CV_16UC1);
    mUDispImage.setTo(0);

    int width=mDisparity.cols;
    int height=mDisparity.rows;
    for(int row=0;row<height;row++)
    {
        auto  pRowInDisp=mDisparity.ptr<uchar>(row);
        for(int col=0;col<width;col++)
        {
            uint8_t currDisp=pRowInDisp[col];
            if(currDisp>0&&currDisp<param.mMaxDisp)
            {
                mUDispImage.at<ushort>(currDisp,col)++;
            }
        }
    }
//    //DeBUG
//    cv::Mat im_Udisp;
//    mUDispImage.convertTo(im_Udisp,CV_8UC1);
//    cv::imshow("imshow U disparity",im_Udisp);
}

void DetectGround::computeVDisparity()
{
    //init v disparity image
    if(mVDispImage.empty())
        mVDispImage.create(mDisparity.rows, param.mMaxDisp, CV_16UC1);
    mVDispImage.setTo(0);
    int width=mDisparity.cols;
    int height=mDisparity.rows;

    for(int row=0;row<height;row++)
    {
        auto  pRowInDisp=mDisparity.ptr<uchar>(row);
        for(int col=0;col<width;col++)
        {
            uint8_t currDisp=pRowInDisp[col];
            if(currDisp>0&&currDisp<param.mMaxDisp)
                mVDispImage.at<ushort>(row,currDisp)++;
        }
    }
    //DeBUG
    cv::Mat im_Vdisp;
    mVDispImage.convertTo(im_Vdisp,CV_8UC1);
    cv::imshow("v disparity",im_Vdisp);
}
bool DetectGround::GetGroundLine() {

    mGroundLine.reserve(param.mMaxDisp);
    mVDispImage.convertTo(mVDispBinary, CV_8UC1);
    //阈值操作
    cv::threshold(mVDispBinary, mVDispBinary, 15, 255, CV_THRESH_TOZERO);

    //高斯滤波
    cv::GaussianBlur(mVDispBinary,mVDispBinary,cv::Size(3,3),param.Gaussian_sigma);

    //Lsd检测
    std::vector<cv::Vec4f> lines_LSD;
    cv::Ptr<cv::ximgproc::FastLineDetector> detector = cv::ximgproc::createFastLineDetector();
    detector->detect(mVDispBinary, lines_LSD);

    if(lines_LSD.empty())
    {
        std::cout<<"NO detect any lines!!!"<<std::endl;
        return false;
    }
    //直线分类
    std::vector<cv::Vec4f> biasLines;//斜线
    for(auto line:lines_LSD)
    {
        if (std::abs(line[0] - line[2])<6)
            continue;
        biasLines.push_back(line);
    }

    //合并倾斜直线
    std::vector<BiasLine> mBiasLines;
    MergeBiasLines(biasLines, mBiasLines);

    //按照权重进行排序
    std::sort(mBiasLines.begin(), mBiasLines.end(), std::greater<BiasLine>());

    //合并后的斜率k和偏移b,使用权重高的
    float k=mBiasLines[0].cur_k;
    float b=mBiasLines[0].cur_b;

    //交点
    for(int currDisp=0;currDisp<param.mMaxDisp;currDisp++)
    {
        float currV=k*currDisp+b;
        mGroundLine[currDisp]=currV+param.mGroundLineBias;
    }
    //Debug
    cv::Mat im_ ;
    cv::cvtColor(mVDispBinary,im_,CV_GRAY2BGR);
    cv::line(im_,cv::Point(mBiasLines[0].x1,mBiasLines[0].y1),cv::Point(mBiasLines[0].x2,mBiasLines[0].y2),cv::Scalar(0,0,255),2);
    cv::imshow("lines",im_);
    return  true;
}
void DetectGround::groundMaskExtraction() {
    int height=mDisparity.rows;
    int width=mDisparity.cols;
    if(mGroundmask.empty())
        mGroundmask.create(mDisparity.size(), CV_8UC1);
    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {
            int currDisp=mDisparity.at<uchar>(i, j);
            if(currDisp>0)
            {
                //5为容忍度
                if(i>=mGroundLine[currDisp]-5)
                    mGroundmask.at<uchar >(i,j)=0;
                else
                    mGroundmask.at<uchar >(i,j)=255;
            }
            else
                mGroundmask.at<uchar >(i,j)=255;
        }
    }
    //Debug
    cv::imshow("ground mask",mGroundmask);
}
void DetectGround::GetSample(std::vector<double>& coeffient, std::vector<cv::Point3f> &d_list) {

    int32_t nums=d_list.size();
    std::vector<int32_t> sample_index;
    // draw 3 measurements
    int32_t k = 0;
    while (sample_index.size() < 3 && k < 1000)
    {
        // draw random measurement
        int32_t curr_index = rand() % nums;

        // first observation
        if (sample_index.empty())
            sample_index.push_back(curr_index);
        // second observation
        else if (sample_index.size() == 1)
        {
            // check distance to first point
            float diff_u = d_list[curr_index].x - d_list[sample_index[0]].x;
            float diff_v = d_list[curr_index].y - d_list[sample_index[0]].y;
            if (std::sqrt(diff_u*diff_u + diff_v*diff_v) > 50)
                sample_index.push_back(curr_index);
        }
        // third observation
        else
        {
            // check distance to line between first and second point
            float vu = d_list[sample_index[1]].x - d_list[sample_index[0]].x;
            float vv = d_list[sample_index[1]].y - d_list[sample_index[0]].y;
            float norm = std::sqrt(vu*vu + vv*vv);
            float nu = +vv / norm;
            float nv = -vu / norm;
            float ru = d_list[curr_index].x - d_list[sample_index[0]].x;
            float rv = d_list[curr_index].y - d_list[sample_index[0]].y;
            if (std::abs(nu*ru + nv*rv) > 50)
                sample_index.push_back(curr_index);
        }
        k++;
    }
    plane_fitting(coeffient,d_list,sample_index);
}
void DetectGround::plane_fitting(std::vector<double>& coeffient, std::vector<cv::Point3f> &input,std::vector<int> &index)
{
    cv::Mat dst = cv::Mat(3, 3, CV_32F, cv::Scalar(0));//初始化系数矩阵A
    cv::Mat out = cv::Mat(3, 1, CV_32F, cv::Scalar(0));//初始化矩阵b
    for (auto i:index)
    {
        //计算3*3的系数矩阵
        dst.at<float>(0, 0) = dst.at<float>(0, 0) + pow(input[i].x, 2);
        dst.at<float>(0, 1) = dst.at<float>(0, 1) + input[i].x*input[i].y;
        dst.at<float>(0, 2) = dst.at<float>(0, 2) + input[i].x;
        dst.at<float>(1, 0) = dst.at<float>(1, 0) + input[i].x*input[i].y;
        dst.at<float>(1, 1) = dst.at<float>(1, 1) + pow(input[i].y, 2);
        dst.at<float>(1, 2) = dst.at<float>(1, 2) + input[i].y;
        dst.at<float>(2, 0) = dst.at<float>(2, 0) + input[i].x;
        dst.at<float>(2, 1) = dst.at<float>(2, 1) + input[i].y;
        dst.at<float>(2, 2) = index.size();
        //计算3*1的结果矩阵
        out.at<float>(0, 0) = out.at<float>(0, 0) + input[i].x*input[i].z;
        out.at<float>(1, 0) = out.at<float>(1, 0) + input[i].y*input[i].z;
        out.at<float>(2, 0) = out.at<float>(2, 0) + input[i].z;
    }
    //判断矩阵是否奇异
    double determ = cv::determinant(dst);
    if (std::abs(determ) < 0.001) {
        std::cout << "矩阵奇异" << std::endl;
        return;
    }
    cv::Mat inv;
    invert(dst, inv);//求矩阵的逆
    cv::Mat output = inv*out;//计算输出
    coeffient.clear();//把结果输出
    coeffient.push_back(output.at<float>(0, 0));
    coeffient.push_back(output.at<float>(1, 0));
    coeffient.push_back(output.at<float>(2, 0));
}

void DetectGround::RansacComputePlane() {
    //rand seed
    std::srand(time(NULL));
    //improve precision
    mDisparity.convertTo(mDispFloat, CV_32FC1);
    //init d list
    std::vector<cv::Point3f> d_list;
    if(!mGroundmask.empty())
    {
        for(int32_t u=0; u < mDisparity.size().width; u+=5)
        {
            for(int32_t v=0; v < mDisparity.size().height; v+=5)
            {
                float d=mDispFloat.at<float>(v,u);
                if(d>=1&&mGroundmask.at<uchar>(v,u)==0)
                {
                    d_list.emplace_back(u,v,d);
                }
            }
        }
    } else{
        int x0 = 200;
        int x1 = mDisparity.cols-200;
        int y0 = mDisparity.rows/2;
        int y1 = mDisparity.rows;
        for(int32_t u=x0; u < x1; u+=5)
        {
            for(int32_t v=y0; v < y1; v+=5)
            {
                float d=mDispFloat.at<float>(v,u);
                if(d>=1&&mGroundmask.at<uchar>(v,u)==0)
                {
                    d_list.emplace_back(u,v,d);
                }
            }
        }

    }
    std::vector<int32_t> curr_inlier;
    std::vector<int32_t> best_inlier;
    //iterator 300 times
    for(int32_t i=0;i<300;i++) {
        //sample
        GetSample(mplane, d_list);
        curr_inlier.clear();
        for (int32_t i = 0; i < d_list.size(); i++)
        {
            if (std::abs(mplane[0] * d_list[i].x + mplane[1] * d_list[i].y + mplane[2] - d_list[i].z) < 1)
                curr_inlier.push_back(i);
        }
        if(curr_inlier.size()>best_inlier.size())
            best_inlier=curr_inlier;
    }
    //reoptimize plane with inliers only
    if(curr_inlier.size()>3)
        plane_fitting(mplane,d_list,best_inlier);
    //std::cout<<"plane: "<<mplane[0]<<" "<<mplane[1]<<" "<<mplane[2]<<std::endl;

    if (mGroundMapWithPlane.empty())
        mGroundMapWithPlane.create(mDisparity.size(), CV_8UC1);

    mGroundMapWithPlane.setTo(255);
    for (int v = 0; v < mDisparity.rows; v++)
    {
        auto* pRowInDisp = mDisparity.ptr<uchar>(v);
        auto* pRowInGndMap = mGroundMapWithPlane.ptr<uchar>(v);
        for (int u = 0; u < mDisparity.cols; u++)
        {
            float currd = pRowInDisp[u];
            if (currd > 0)
            {
                float expectground = mplane[0] * float(u) + float(v)*mplane[1] + mplane[2];
                if (std::abs(currd - expectground) < 1 )
                    pRowInGndMap[u] = 0;
            }
        }
    }
    cv::imshow("mGroundMapWithPlane",mGroundMapWithPlane);
}
void DetectGround::groundPlaneRefinement()
{
    //find Contour
    std::vector<std::vector<cv::Point> > contours0;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat contourImg(mGroundMapWithPlane.rows, mGroundMapWithPlane.cols, CV_8UC1, cv::Scalar(0));
    cv:: Mat contourSrc;

    //取反
    contourSrc = 255 - mGroundMapWithPlane;

    findContours(contourSrc, contours0, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int idx = 0;
    int maxarea = -1;
    int maxidx = -1;
    //hierarchy[idx][0]表示后一个轮廓的idx,否则为－１
    for (; idx >= 0; idx = hierarchy[idx][0])
    {
        int currerea = contourArea(contours0[idx]);
        if (currerea > maxarea)
        {
            maxidx = idx;
            maxarea = currerea;
        }
    }
    //最大面积轮廓
    drawContours(contourImg, contours0, maxidx, 255,-1);

    mGroundUpperBound = std::vector<float>(mGroundMapWithPlane.cols, 0);
    for (int u = 0; u < contourImg.cols; u++)
    {
        mGroundUpperBound[u] = FLT_MAX;
        for (int v = 0; v < contourImg.rows; v++)
        {
            if (contourImg.at<uchar>(v, u) == 255 && v < mGroundUpperBound[u])
                mGroundUpperBound[u] = v;
        }
    }
    for (int u = 0; u < contourImg.cols; u++)
    {
        if (mGroundUpperBound[u] != FLT_MAX)
            for (int v = contourImg.rows - 1; v >= mGroundUpperBound[u]; v--)
                contourImg.at<uchar>(v, u) = 255;
    }
    mGroundMapWithPlane = cv::Scalar(255,255,255) - contourImg;
	cv::Mat temp;
	cvtColor(mGroundMapWithPlane, temp, CV_GRAY2BGR);
	for (int i = 0; i < contourImg.cols; i++)
	{
		circle(temp, cv::Point(i, mGroundUpperBound[i]), 2, cv::Scalar(255, 255, 0), -1);
	}
	imshow("refine with line", temp);
}
void DetectGround::showGroundWithImage()
{
    static int num=0;
    cv::Scalar color=cv::Scalar(0,255,0);
    cv::Mat result(mDisparity.size(), CV_8UC3);
    cv::Mat mask(mDisparity.size(),CV_8UC3,cv::Scalar(0,0,0));
    cv::Mat LeftImgBGR(mDisparity.size(), CV_8UC3);
    mask.setTo(color, mGroundMapWithPlane==0);
//    cv::imwrite("../mask/mask_"+cv::format("%010d.png",num++),mask);
    addWeighted(mask, 0.3, left, 0.7,0,result);
    cv::imshow("mask",result);
}
void DetectGround::calculate_line(cv::Vec4f &line,float &k, float &b)
{
    cv::Point2f Pt1(line[0],line[1]);
    cv::Point2f Pt2(line[2],line[3]);
    if(Pt1.x>Pt2.x)
        cv::swap(Pt1,Pt2);
    k = (Pt1.y - Pt2.y) / (Pt1.x - Pt2.x);
    b = (Pt2.y * Pt1.x - Pt1.y *Pt2.x) / (Pt1.x - Pt2.x);
}
void DetectGround::MergeBiasLines(std::vector<cv::Vec4f> &lines,std::vector<BiasLine> &cLines)
{
    //遍历线段
    for(auto line:lines)
    {
        //获取线段端点值
        cv::Point2f Pt1(line[0],line[1]);
        cv::Point2f Pt2(line[2],line[3]);

        if(Pt1.x>Pt2.x)
            cv::swap(Pt1,Pt2);
        //计算权重
        double weight =std::sqrt(std::pow(Pt1.x-Pt2.x,2)+std::pow(Pt1.y-Pt2.y,2));

        if(weight!=0&&weight>param.min_weight)
        {
            //计算k与b
            float k,b;
            calculate_line(line,k,b);
            //初始化
            if(cLines.empty()) {
                BiasLine temp;
                temp.cur_k=k;
                temp.cur_b=b;
                temp.k_sum=k*weight;
                temp.b_sum=b*weight;
                temp.weight_sum=weight;
                temp.x1=Pt1.x;
                temp.y1=Pt1.y;
                temp.x2=Pt2.x;
                temp.y2=Pt2.y;
                cLines.push_back(temp);
                continue;
            }
            //根据k的差异做加权
            //首先获取cLines数组里面k距离最近的那个
            double min_k=std::numeric_limits<double>::max();//初始化
            int min_index=-1;
            for(int i=0;i<cLines.size();i++)
            {
                double neighbor_k=std::abs(cLines[i].cur_k-k);
                if(neighbor_k<min_k)
                {
                    min_k=neighbor_k;
                    min_index=i;
                }
            }
            BiasLine &neighbor_line=cLines[min_index];
            //小于最大k差值，认为是同一条线
            if(std::abs(neighbor_line.cur_k-k)<param.max_k_distance)
            {
                neighbor_line.weight_sum+=weight;
                neighbor_line.k_sum+=k*weight;
                neighbor_line.b_sum+=b*weight;
                neighbor_line.cur_k=neighbor_line.k_sum/neighbor_line.weight_sum;
                neighbor_line.cur_b=neighbor_line.b_sum/neighbor_line.weight_sum;

                //最左上角
                if(neighbor_line.x1>Pt1.x)
                {
                    neighbor_line.x1=Pt1.x;
                    neighbor_line.y1=Pt1.y;
                }
                //最右下角
                if(neighbor_line.x2<Pt2.x)
                {
                    neighbor_line.x2=Pt2.x;
                    neighbor_line.y2=Pt2.y;
                }
            }
            else
            {
                BiasLine cline;
                cline.cur_k=k;
                cline.cur_b=b;
                cline.k_sum=k*weight;
                cline.b_sum=b*weight;
                cline.weight_sum=weight;
                cline.x1=Pt1.x;
                cline.y1=Pt1.y;
                cline.x2=Pt2.x;
                cline.y2=Pt2.y;
                cLines.push_back(cline);
            }
        }
    }
}