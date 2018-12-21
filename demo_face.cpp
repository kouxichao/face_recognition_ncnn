#include<stdio.h>
#include "face_recognization.h"

int main(int argc, char* argv[])
{
    int  id, flag, count;
    DKSSingleDetectionRes box[1];
    box[0].box = {0,0,112,0,112,112,0,112};
    DKSMultiDetectionRes boxes;
    boxes.num = 1;
    boxes.boxes[0] = box[0];
    DKSFaceRegisterParam rgp;
    rgp.index = 0;
    DKSFaceRecognizationParam rcp;
    rcp.index = 0;
    rcp.threshold = 0.5;
    rcp.k = 5;
    
    //注册
    if(*(argv[1]) == '0')
    {
        DKFaceRegisterInit();
        int count = *(argv[2]) - 48;
        for(int i = 0; i < count; i++)
        {
            char*   rgbfilename = argv[2+i+1];
            DKFaceRegisterProcess(rgbfilename, 100, 100, boxes, rgp);//示例中没有用到iWidth,iHeight两个参数。
            DKFaceRegisterEnd(count - (i+1) ? 1 : 0, i+1);
        }
    }

    //识别
    if(*(argv[1]) == '1')
    {
        char*   rgbfilename = argv[2];
        DKFaceRecognizationInit();
        id = DKFaceRecognizationProcess(rgbfilename, 100, 100, boxes, rcp);//示例中没有用到100,100两个参数。
        DKFaceRecognizationEnd();
        printf("ID:%d\n", id);
    }
    return 0;
}
