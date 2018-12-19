#include<stdio.h>
#include<unistd.h>
#include<iostream>
#include "face_recognization.h"
#include<cstring>
int main(int argc, char* argv[])
{
    char *root_dir = argv[2];
    
    char bbox_path[50];
    strcpy(bbox_path, root_dir);
    strcat(bbox_path, "bbox.xy"); 
    FILE *fp = fopen(bbox_path, "r+");
    if(NULL == fp)
    {
	    fprintf(stderr, "fopen bbox.xy error\n");
    }
    int  id;

    DKSSingleDetectionRes box[1];
    DKSMultiDetectionRes boxes;
    DKSFaceRegisterParam rgp;
    rgp.index = 0;
    DKSFaceRecognizationParam rcp;
    rcp.index = 0;
    rcp.threshold = 0.3;
    char pre_name[50] = {};

    //注册
    if(*(argv[1]) == '0')
    {
        FILE* fp_idx_name = fopen("idx_name", "a+");
        if(NULL == fp_idx_name)
        {
	        fprintf(stderr, "fopen idx_name error\n");
        }
//	    std::vector<char*> idx_name;
        char name[50];
        char idx[5];
        int right,left,bottom,top;
        DKFaceRegisterInit();
        while(1)
        {

            if((fscanf(fp, "%s %s %d,%d,%d,%d", name, idx, &right, &left, &bottom, &top)) == EOF)
	        {
			    fprintf(stderr, "fscanf end(error)\n");
                break;
            }

            fprintf(stderr, "name : %s\n", name);
	        if(strstr(name, "test") == NULL)
            {
                std::string rgbfilename = std::string(root_dir) + std::string(name) + \
                 '/' + "support/" + name + "_" + idx;
                printf("PATH: %s\n", rgbfilename.data()); 
                if(access((rgbfilename + std::string(".jpg")).data(), 0) == 0)
                    rgbfilename = rgbfilename + std::string(".jpg");
                else
                    rgbfilename = rgbfilename + std::string(".png");

                box[0].box = {left,top,right,top,right,bottom,left,bottom};
//    box[0].box = {0,0,112,0,112,112,0,112};

                boxes.num = 1;
                boxes.boxes[0] = box[0];

                DKFaceRegisterProcess((char*)rgbfilename.data(), 100, 100, boxes, rgp);//示例中没有用到iWidth,iHeight两个参数。
                if(strcmp(pre_name, name) == 0)
                    DKFaceRegisterEnd(1, 2); //第二个参数大于1小于10，表示直接插入最后的记录中。
                else
                {
//                    idx_name.push_back(name);
                    fprintf(fp_idx_name, "%s\n", name);
                    DKFaceRegisterEnd(1, 1); //第二个参数等于1,表示增加新的记录。
                }
                strcpy(pre_name,  name);
            }
        }
        fclose(fp_idx_name);

    }

    //识别
    if(*(argv[1]) == '1')
    {
        FILE* fp_idx_name = fopen("idx_name", "r+");
        if(NULL == fp_idx_name)
        {
	        fprintf(stderr, "fopen idx_name error\n");
        }
        char idx_name[20][50];
        int index = 0;
        while(fscanf(fp_idx_name, "%s", idx_name[index]) != EOF)
        {
//            fprintf(stderr, "temp_name : %s\n", idx_name[index]);
            index++;
        }
//        fprintf(stderr, "temp_name : %s\n", idx_name[0]);

        DKFaceRecognizationInit();
        char name[30];
        char idx[5];
        int right,left,bottom,top;
        float num = 0, acc = 0;
        while(1)
        {

            if((fscanf(fp, "%s %s %d,%d,%d,%d", name, idx, &right, &left, &bottom, &top)) == EOF)
	        {
		        fprintf(stderr, "fscanf end(error)\n");
                break;
            }

//            fprintf(stderr, "name : %s\n", name);
	        if(strstr(name, "test") != NULL && strcmp(name, "xiena_test") == 0)
            {
//                fprintf(stderr, "ori_pos:%d_%d_%d_%d\n", right, left, top, bottom);

                std::size_t found = std::string(name).find_last_of("_");
                std::string rea_name = std::string(name).substr(0, found);  
                std::string rgbfilename = std::string(root_dir) + rea_name + "/" + "test/" + name + "_" + idx;
                
                if(access((rgbfilename + std::string(".jpg")).data(), 0) == 0)
                    rgbfilename = rgbfilename + std::string(".jpg");
                else
                    rgbfilename = rgbfilename + std::string(".png");
                //
                box[0].box = {left,top,right,top,right,bottom,left,bottom};
                boxes.num = 1;
                boxes.boxes[0] = box[0];
//                printf("PATH: %s\n", rgbfilename.data()); 

        	    id = DKFaceRecognizationProcess((char*)rgbfilename.data(), 100, 100, boxes, rcp);//示例中没有用到100,100两个参数。
        	    printf("image:%s \t gt_name:%s \t pre_name(ID):%s_(%d)\n", \
                (std::string(name) + "_" + idx).data(), rea_name.data(), idx_name[id], id);
                if((strcmp(rea_name.data(), idx_name[id])) == 0)
                    acc++;
                num++;
            }
	    }

       	DKFaceRecognizationEnd();
        fclose(fp_idx_name);
                        printf("num:%d \t acc:%d \n", num, acc);

        float accuracy = acc / num;
        fprintf(stderr, "accuracy: %.2f%%\n", accuracy);
	}

    fclose(fp);
    return 0;
}
