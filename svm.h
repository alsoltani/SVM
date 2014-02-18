#ifndef _SVM_H
#define _SVM_H

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

using namespace cv;
using namespace std;

class SVMModel{
public:
	SVMModel();
	CvSVM svm;
	
	void SVM_ConsoleVersion(OpenAll);
	void SVM_Train_Custom(OpenTrain, CvSVMParams);
	void SVM_Train_Optimal(OpenTrain, CvParamGrid, CvParamGrid, CvSVMParams, int);
	void SVM_Test(OpenTrain, OpenTest);
	void PrintParamGrid(CvSVMParams, CvParamGrid, CvParamGrid);
	void PrintParam(CvSVMParams);
};
#endif