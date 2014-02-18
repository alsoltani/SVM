#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <fstream>
#include <string>
#include <sstream>
#include <windows.h>
#include <iterator>
#include <ctime>

#include "save_file.h"

Save::Save(){}

void Save::SaveMatrix(string Output_Directory, string Filename, Mat Matrix){
	stringstream ss;
	ss << Output_Directory << "\\" << Filename << ".txt";
	FileStorage fs(ss.str(), cv::FileStorage::WRITE);
	fs << Filename << Matrix;
}


using namespace cv;
using namespace std;

