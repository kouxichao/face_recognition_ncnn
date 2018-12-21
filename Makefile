debug =
MODE = 
CFLAGS += -I$(PWD) \
	-I$(PWD)/dlib \
    -I$(PWD)/ncnn \
	-L$(PWD)/lib -ldlib -lncnn -ldl -pthread -lX11
CFLAGS += $(MODE) $(debug) -O3 -fopenmp -std=c++11 -lsqlite3 -Wall -DDLIB_JPEG_SUPPORT -DDLIB_PNG_SUPPORT 
#SRCS := $(wildcard *.cpp)
SRCS := interface_face.cpp 

ifeq ($(MODE), -DJPG_DEMO)
all:interface_face evaluate detect_face demo_face
else 
all:interface_face
endif

detect_face: 
	g++ -o detect_face detect_face.cpp $(CFLAGS)   
	
interface_face:
	g++ $(SRCS) $(CFLAGS)  -c -o interface_face.cpp.o 
	ar r libface.a interface_face.cpp.o

evaluate:interface_face
	g++  evaluate.cpp    libface.a $(CFLAGS)  -o evaluate 

demo_face:interface_face
	g++ demo_face.cpp  libface.a $(CFLAGS) -o demo_face 
