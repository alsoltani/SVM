#ifndef _SAVE_FILE_H
#define _SAVE_FILE_H

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

using namespace cv;
using namespace std;

class Save{
public :
	Save();
	void SaveMatrix(string, string, Mat);
};
#endif