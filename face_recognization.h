#ifndef FACE_RECOGNIZATION_h
#define FACE_RECOGNIZATION_h
#include "net.h"

#ifndef DKMAXBOXNUM
#define DKMAXBOXNUM 30 // 最多可检测对象数目
#endif

#ifndef DKMINFACEREGISTERIMGNUM
#define DKMINFACEREGISTERIMGNUM  5 // 人脸注册时最少需采集人脸数目
#endif

typedef struct
{
    // 顺时针排列
    int x1; // 偏左上角横坐标
    int y1; // 偏左上角纵坐标
    int x2; // 偏右上角横坐标
    int y2; // 偏右上角纵坐标
    int x3; // 偏右下角横坐标
    int y3; // 偏右下角纵坐标
    int x4; // 偏左下角横坐标
    int y4; // 偏左下角纵坐标
}DKSBox;

typedef struct
{
    int tag; // 标签，与宏定义对应
    float confidence; // 置信度
    DKSBox box;
}DKSSingleDetectionRes;

typedef struct
{
    int num; // 当前检测出的物体总数目
    DKSSingleDetectionRes boxes[DKMAXBOXNUM];
}DKSMultiDetectionRes;

typedef struct
{
    int index; //边界框索引
    float threshold; // 相似度阈值，相似度超过该值认为不能认出该人脸
}DKSFaceRecognizationParam;

typedef struct
{
    int index; //边界框索引
}DKSFaceRegisterParam;

/*

工具函数

*/

//计算向量内积
float dot(float* fc1, float* fc2);

//规范化向量
int normalize(ncnn::Mat& fc1);

//knn最近邻实现
int knn(std::vector< std::pair<int, float> >& re, int k);


/*

人脸注册识别相关函数

*/

// 说明：录入人脸，在学习人脸阶段，获取至少DKMINFACEREGISTERIMGNUM张同一人脸图片
// 初始化，连接sqlite，准备写入人脸特征和语音，初始化学习图片次数为0
void DKFaceRegisterInit();

// 根据检测到的人脸结果计算特征
int DKFaceRegisterProcess(char * imgfilename, int iWidth, int iHeight, DKSMultiDetectionRes boxes, DKSFaceRegisterParam param);

// 将计算好的特征存入sqlite中（flag为1），或取消学习人脸（flag为0）
void DKFaceRegisterEnd(int flag, int count);

// 说明：从已注册的人脸中识别对应的人脸
// 初始化，连接sqlite，获取人脸库中各个人脸的特征
void DKFaceRecognizationInit();

// �，如果没有识别出或相似度大于柨识别参数中定义），则输出null，否则输出识别出的人的index。
int DKFaceRecognizationProcess(char * imgfilename, int iWidth, int iHeight, DKSMultiDetectionRes boxes, DKSFaceRecognizationParam param);

// 释放人脸识别资源
void DKFaceRecognizationEnd();

#endif 
