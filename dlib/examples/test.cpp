#include "dlib/dlib/image_processing/frontal_face_detector.h"
#include "dlib/dlib/image_processing.h"
#include "dlib/dlib/image_io.h"
#include "iostream"
#include "dlib/dlib/clustering.h"
#include "dlib/dlib/gui_widgets.h"
using namespace dlib;
using namespace std;

int main(int argc, char** argv)
{
 
	try
	{
		frontal_face_detector detector = get_frontal_face_detector();

		shape_predictor sp;
		deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
 
 
                cout << "processing image " << argv[1] << endl;
		array2d<rgb_pixel> img;
		load_image(img, argv[1]);
//		pyramid_up(img);
                image_window win(img); 
	
         	std::vector<rectangle> dets = detector(img);
 
			// Now we will go ask the shape_predictor to tell us the pose of
			// each face we detected.
         	//std::vector<full_object_detection> shapes;
                float index = 0, area = 0; 
		for (unsigned long j = 0; j < dets.size(); ++j)
		{
                    float width = dets[j].right() - dets[j].left();
                    float height = dets[j].bottom() - dets[j].top();
                    if(width * height > area)
                    {
                        index = j;
                        area = width * height;
                    }
                    
		}
                             
                        
                full_object_detection shape = sp(img, dets[index]);
	   //     dlib::array<array2d<rgb_pixel> > face_chips;
                matrix<rgb_pixel> face_chips;
		extract_image_chip(img, get_face_chip_details(shape), face_chips);
                win.add_overlay(dets[index]);
		cout << "image process finished" << endl;
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
        return 0;
}

