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

#include "open_file.h"
#include "save_file.h"
#include "svm.h"

using namespace cv;
using namespace std;

int main(){
	OpenAll OA;
	SVMModel SVMM;
	OA.Open_ConsoleVersion();
	SVMM.SVM_ConsoleVersion(OA);
	return 0;
	system("PAUSE");
}