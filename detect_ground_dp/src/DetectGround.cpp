//
// Created by hexi on 2020/4/15.
//
#include <DetectGround.h>
#include <opencv2/line_descriptor/descriptor.hpp>
#include "opencv2/ximgproc.hpp"
#include <cmath>
#include <Util.h>
double min_weight=20;
double max_k_distance=0.5;

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

void calculate_line(cv::Vec4f &line,float &k, float &b)
{
    cv::Point2f Pt1(line[0],line[1]);
    cv::Point2f Pt2(line[2],line[3]);
    if(Pt1.x>Pt2.x)
        cv::swap(Pt1,Pt2);
    k = (Pt1.y - Pt2.y) / (Pt1.x - Pt2.x);
    b = (Pt2.y * Pt1.x - Pt1.y *Pt2.x) / (Pt1.x - Pt2.x);
}
void MergeBiasLines(std::vector<cv::Vec4f> &lines,std::vector<BiasLine> &cLines)
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

        if(weight!=0&&weight>min_weight)
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
            if(std::abs(neighbor_line.cur_k-k)<max_k_distance)
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

DetectGround::DetectGround(DetectGround::parameter param,cv::Mat &left,cv::Mat &right):left(left),right(right) {
    this->param=param;
    mStereoSGBM=cv::StereoSGBM::create(0,param.mMaxDisp,param.mWinSize,param.mP1,param.mP2,0,param.mPreFilterCap,5,0,0,cv::StereoSGBM::MODE_HH);
}
DetectGround::~DetectGround() {

}

//计算视差图像
void DetectGround::computeDisparity() {
    //compute disparity
    cv::Mat disp16s;
    mStereoSGBM->compute(left,right,disp16s);
    disp16s=disp16s/16;
    disp16s.convertTo(disp,CV_8UC1);
    mDispWithoutGround=disp.clone();
//    //DEBUG
//    cv::imshow("disparity",disp);
}
void DetectGround::computeVDisparity()
{
    //init v disparity image
    if(mVDispImage.empty()) {
        mVDispImage.create(disp.rows, param.mMaxDisp, CV_8UC1);
    }
    //set 0
    mVDispImage.setTo(0);

    int width=disp.cols;
    int height=disp.rows;
    for(int row=0;row<height;row++)
    {
        auto  pRowInDisp=disp.ptr<uchar>(row);
        for(int col=0;col<width;col++)
        {
            uint8_t currDisp=pRowInDisp[col];
            if(currDisp>0&&currDisp<param.mMaxDisp)
                mVDispImage.at<uchar>(row,currDisp)++;
        }
    }
    //DeBUG
//    cv::imshow("v disparity",mVDispImage);
}

void DetectGround::Detect() {

    //消除一些离散噪点
    cv::threshold(mVDispImage,mVDispBinary,10,255,cv::THRESH_TOZERO);

    //Lsd Detect Lines
    std::vector<cv::Vec4f> lines_LSD;
    cv::Ptr<cv::ximgproc::FastLineDetector> detector = cv::ximgproc::createFastLineDetector();
    detector->detect(mVDispBinary, lines_LSD);

    //直线分类
    std::vector<cv::Vec4f> biasLines;//斜线
    std::vector<cv::Vec4f> verticalLines;//垂直直线
    for(auto line:lines_LSD)
    {
        if (std::abs(line[0] - line[2])<5)
            verticalLines.push_back(line);
        else
            biasLines.push_back(line);
    }

    //合并倾斜直线
    std::vector<BiasLine> mBiasLines;
    MergeBiasLines(biasLines, mBiasLines);

    //按照权重进行排序
    std::sort(mBiasLines.begin(), mBiasLines.end(), std::greater<BiasLine>());

    //合并后的斜率k和偏移b,使用权重高的
    float k=mBiasLines[0].cur_k,b=mBiasLines[0].cur_b;

    for(int i=0;i<mDispWithoutGround.rows;i++)
    {
        for(int j=0;j<mDispWithoutGround.cols;j++)
        {
            if(mDispWithoutGround.at<uchar>(i,j)<((double)i-b)/k+8)
                mDispWithoutGround.at<uchar >(i,j)=0;
        }
    }
//    cv::imshow("mDispWithoutGround",mDispWithoutGround);

    //计算u视差图,b去除天空,远点
    computeUDisparity(b);

    cv::Mat sum,dir;
    cv::threshold(mUDispImage,mUDispImage,15,255,CV_THRESH_TOZERO);
    DpOptimize(sum,dir,mUDispImage);

    cv::Mat show=cv::Mat::zeros(mUDispImage.size(),CV_8UC3);
    float pre=k*points[0].y+b;
    for(int i=1;i<points.size();i++)
    {

        float y=k*points[i].y+b;
        if(y>=left.rows)
            y=left.rows-2;

        cv::line(left,cv::Point(points[i-1].x,pre),cv::Point(points[i].x,y),cv::Scalar(255,0,0),2);
        cv::line(show,points[i],points[i],cv::Scalar(255,0,0));
        pre=y;
    }

    cv::imshow("left",left);
    cv::imshow("show",show);

    //二值化
    cv::threshold(mUDispImage,mUDispBinary,10,255,CV_THRESH_BINARY);
    cv::imshow("mUDispImage",mUDispBinary);
}
//compute U disparity image
void DetectGround::computeUDisparity(float b=0) {
    //init u disparity image
    if(mUDispImage.empty())
        mUDispImage.create(param.mMaxDisp,mDispWithoutGround.cols,CV_8UC1);
    mUDispImage.setTo(0);
    int width=mDispWithoutGround.cols;
    int height=mDispWithoutGround.rows;
    for(int row=b;row<height;row++)
    {
        auto  pRowInDisp=mDispWithoutGround.ptr<uchar>(row);
        for(int col=0;col<width;col++)
        {
            uint8_t currDisp=pRowInDisp[col];
            if(currDisp>0&&currDisp<param.mMaxDisp)
                mUDispImage.at<uchar>(currDisp,col)++;
        }
    }
//    DeBUG
    cv::imshow("imshow U disparity",mUDispImage);
}
void DetectGround::DpOptimize(cv::Mat cost ,cv::Mat dir,cv::Mat Udisparity) {
    float lamda=30;
    cost=cv::Mat::zeros(Udisparity.size(),CV_32F);
    dir=cv::Mat::zeros(Udisparity.size(),CV_32SC1);
    int Umax=Udisparity.cols;
    int Dmax=Udisparity.rows;
    //初始化第一列
    for(int i=0;i<Dmax;i++)
    {
        cost.at<float>(i,0)=-(float)Udisparity.at<uchar>(i,0);
        dir.at<int>(i,0)=-1;

    }

    for(int u=1;u<Umax;u++)
    {
        for(int d=0;d<Dmax;d++)
        {
            for(int p=0;p<Dmax;p++)
            {
                float temp=cost.at<float>(p,u-1)-(float)Udisparity.at<uchar>(d,u)+lamda*(float)std::abs(d-p);
                cost.at<float>(d,u)=std::min(cost.at<float>(d,u),temp);
                if(cost.at<float>(d,u)==temp)
                    dir.at<int>(d,u)=p;
            }
        }
    }
    float MinCost=FLT_MAX;
    float MinIndex=-1;
    for(int i=0;i<Dmax;i++)
    {
        if(MinCost>cost.at<float>(i,Umax-1))
        {
            MinCost=cost.at<float>(i,Umax-1);
            MinIndex=i;
        }
    }
    int i=MinIndex;
    for(int j=Umax-2;j>=0;j--)
    {
        i=dir.at<int>(i,j);
        points.push_back(cv::Point(j,i));
    }
}



