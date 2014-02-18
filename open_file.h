#ifndef _OPEN_FILE_H
#define _OPEN_FILE_H

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

using namespace cv;
using namespace std;


class OpenTrain{
public:
	OpenTrain();
	int Count = 0; //To count elements from 1st class
	vector<string> Indexes; //Indexes of all images to open

	int Nb_Data_First_Class = Count;
	int Nb_Data_Second_Class = Indexes.size() - Count;
	int Nb_Files = Indexes.size();

	Mat Data; //Data will contain the dataset,
	Mat Labels; // Labels the class values.

	void OpenText(string, string, string);
	void OpenImages(string, string, int, int, int, int);
	void Open(string, string, string, string, string, int, int, int, int);
	
};

class OpenTest{
public:
	OpenTest();

	int Nb_Files = Indexes.size();

	Mat Data;
	Mat Labels;

	void OpenText(string, string);
	void OpenImages(string, string, int, int, int);
	void Open(string, string, string, string, int, int, int);

private:
	vector<string> Indexes;
};

class OpenAll{
public :
	OpenAll();
	string Directory_Text_Files;
	string Directory_Images_Files;
	string Format;
	
	int Width_Zone;
	int Height_Zone;

	string File_Train_1;
	string File_Train_2;
	string File_Test;

	int LabVal_1;//Label values for training phase
	int LabVal_2;
	int LabVal_Test; //Label value for testing phase

	OpenTrain OTr;
	OpenTest OTe;

	
	void Open_ConsoleVersion();


};

#endif