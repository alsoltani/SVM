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

#include "utility.h"


float GetLine_Float(){
	float f;
	string Input;
	getline(cin, Input);
	istringstream ss(Input);
	ss >> f;
	return f;
}

int GetLine_Int(){
	int i;
	string Input;
	getline(cin, Input);
	istringstream ss(Input);
	ss >> i;
	return i;
}