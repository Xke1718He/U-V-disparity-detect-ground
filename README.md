# U-V-disparity-detect-ground
This repository uses RANSAC-based, dp-based methods to detect road surfaces using U-V disparity.
## paper
1. introduce U-V disparity
* A Complete U-V-Disparity Study for Stereovision Based 3D Driving Environment analysis [2005] 
* 基于 U-V 视差算法的障碍物识别技术研究[2011]
* U-VDisparityAnalysisinUrbanEnvironments[2011]
## based on the following libraries
* OpenCV3.1+
* cmake 3.15(you can modify cmake version info)
## build the program
* mkdir build 
* cd build
* cmake ..
* make
## related blog
* [my blog](https://blog.csdn.net/He3he3he/article/details/105542815)
## results
### Ransac
  <p align="center">
  <img src="result/detect_ground.gif"/>
  </p>
  
  <p align="center">
  <img src="result/mask.png"/>
  </p>
