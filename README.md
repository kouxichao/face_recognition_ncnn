# face recognition using ncnn framework

编译:
```
make -DJPG_DEMO [-DDEBUG] //选项-DDEBUG可选，会输出相似值等信息
该命令生成demo_face, detect_face，evaluate两个可执行文件。

```

生成脸部检测框坐标文件bbox.xy（如项目中文件bbox.xy,并将其放入${image_dir}目录下）:
```
./detect_face  ${image_dir}  //坐标顺序 right, left, top, bottom 。${image_dir}中image的结构如face_test_images中一样。

注：此时 ${image_dir}文件夹下不应该有bbox.xy。
```

脸部特征存储：
```
./evaluate 0   ${image_dir}  //${image_dir} 必须包含做后一个正斜线/，如./face_test_images/ 

该命令会生成一个idx_name文本文件， 包含数据库脸部特征条目的名称。
```
评估：
```
./evaluate 1  ${image_dir} //打印出测试结果
```

单张测试：
```
demo_face 1 ${one_image}

```

结果示例:
```
result.txt
```
