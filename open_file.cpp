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

OpenTrain::OpenTrain(){}
OpenTest::OpenTest(){}
OpenAll::OpenAll(){}

void OpenTrain::OpenText(string Directory_Text_Files, string File_Train_1, string File_Train_2){

	//Files are converted from ifstream to string format.
	//Count represents size of class 1.

	stringstream ss1;
	ss1 << Directory_Text_Files << "\\" << File_Train_1 << ".txt";

	std::ifstream ifs_1(ss1.str());
	std::string str_1((std::istreambuf_iterator<char>(ifs_1)), std::istreambuf_iterator<char>());

	//We append each 6-digit value at the end of indexes vector.
	//We erase the newline symbol.

	for (int i = 0; 6 * i + 5 < str_1.length(); i++){
		Indexes.push_back(str_1.substr(6 * i, 6));
		str_1.erase(str_1.begin() + 6 * i + 6);
		Count++;
	}
	stringstream ss2;
	ss2 << Directory_Text_Files << "\\" << File_Train_2 << ".txt";

	std::ifstream ifs_2(ss2.str());
	std::string str_2((std::istreambuf_iterator<char>(ifs_2)), std::istreambuf_iterator<char>());

	for (int i = 0; 6 * i + 5 < str_2.length(); i++){
		Indexes.push_back(str_2.substr(6 * i, 6));
		str_2.erase(str_2.begin() + 6 * i + 6);
	}

}

void OpenTrain::OpenImages(string Directory_Images_Files, string Format, 
	int Width_Zone, int Height_Zone, 
	int LabVal_1, int LabVal_2){

	Nb_Data_First_Class = Count;
	Nb_Data_Second_Class = Indexes.size() - Count;
	Nb_Files = Indexes.size();

	int TotalSz_Zone = Width_Zone*Height_Zone;
	Data = Mat::zeros(Nb_Files, TotalSz_Zone, CV_32FC1);
	Labels = Mat::zeros(Nb_Files, 1, CV_32FC1);

	//--------------------------------------Training Set Transformation : 1st class-------------------------------------

	for (int i = 0; i < Nb_Data_First_Class; i++)
	{
		//Opening each file ending with 6-digit_value.jpg, given by IC.indexes[i]
		stringstream ss;
		ss << Directory_Images_Files << "\\" << Indexes[i] << Format;
		Mat Image = cv::imread(ss.str(), 0);

		if (!Image.data) {
			cout << "Error occured while charging image " << Indexes[i] << Format << ".\n" << endl;
			cout << "Please check path or file format.\n\n" << endl;
			system("PAUSE");
		}

		//Resizing to zone selected
		cv::resize(Image, Image, Size(Width_Zone, Height_Zone));

		//Adding each image to a Data row
		int ii = 0; // Column in Mat Data
		for (int l = 0; l < Image.rows; l++) {
			for (int j = 0; j < Image.cols; j++) {
				Data.at<float>(i, ii) = Image.at<uchar>(l, j);
				ii++;
			}
		}
		Labels.at<float>(i, 0) = LabVal_1;

	}

	//--------------------------------------Training Set Transformation : 2nd class--------------------------------------
	for (int i = Nb_Data_First_Class; i < Nb_Files; ++i)
	{
		stringstream ss;
		ss << Directory_Images_Files << "\\" << Indexes[i] << Format;
		Mat Image = cv::imread(ss.str(), 0);

		if (!Image.data) {
			cout << "Error occured while charging image " << Indexes[i] << Format << ".\n" << endl;
			cout << "Please check path or file format.\n\n" << endl;

		}
		cv::resize(Image, Image, Size(Width_Zone, Height_Zone));

		int ii = 0;
		for (int l = 0; l < Image.rows; l++) {
			for (int j = 0; j < Image.cols; j++) {
				Data.at<float>(i, ii) = Image.at<uchar>(l, j);
				ii++;

			}
		}
		Labels.at<float>(i, 0) = LabVal_2;
	}
}

void OpenTrain::Open(string Directory_Text_Files, string Directory_Images_Files, string Format,
	string File_Train_1, string File_Train_2,
	int Width_Zone, int Height_Zone,
	int LabVal_1, int LabVal_2){

	cout << "Charging training data..." << endl;
	OpenTrain::OpenText(Directory_Text_Files, File_Train_1, File_Train_2);
	OpenTrain::OpenImages(Directory_Images_Files, Format, Width_Zone, Height_Zone, LabVal_1, LabVal_2);
	cout << "Training data charged.\n" << endl;
}

//Version of Open, that only uses console for input values.
void OpenAll::Open_ConsoleVersion(){

	cout << "========================================" << endl;
	cout << "    SVM IMAGE CLASSIFICATION PROGRAM." << endl;
	cout << "========================================" << endl;
	cout << "----------------------------------------" << endl;
	cout << "Common directories and parameters" << endl;
	cout << "----------------------------------------" << endl;
	cout << "1) Please enter directory for text files :" << endl;
	getline(cin,Directory_Text_Files);
	cout << "2) Please enter directory for images files :" << endl;
	getline(cin,Directory_Images_Files);
	cout << "3) Please enter file format : " << endl;
	getline(cin, Format);
	cout << "4) Please enter convenient values (width, height) for the image zone." << endl;
	Width_Zone=GetLine_Int();
	Height_Zone = GetLine_Int();

	cout << "----------------------------------------" << endl;
	cout << "Opening training files" << endl;
	cout << "----------------------------------------" << endl;
	cout << "1) Please enter names of the two files you wish to train your SVM on :" << endl;
	getline(cin,File_Train_1);
	getline(cin, File_Train_2);
	
	cout << "2) Please enter the label values for Files 1 and 2." << endl;
	LabVal_1 = GetLine_Int();
	LabVal_2 = GetLine_Int();

	OTr.Open(Directory_Text_Files, Directory_Images_Files, Format,
		File_Train_1, File_Train_2,
		Width_Zone, Height_Zone,
		LabVal_1, LabVal_2);

	cout << "----------------------------------------" << endl;
	cout << "Opening testing files" << endl;
	cout << "----------------------------------------" << endl;
	cout << "1) Please enter name of the file used in testing procedure." << endl;
	getline(cin, File_Test);
	cout << "2) Finally, please enter label of tested file." << endl;
	LabVal_Test = GetLine_Int();

	OTe.Open(Directory_Text_Files, Directory_Images_Files, Format,
		File_Test,
		Width_Zone, Height_Zone,
		LabVal_Test);
}


void OpenTest::OpenText(string Directory_Text_Files, string File_Test){

	stringstream ss;
	ss << Directory_Text_Files << "\\" << File_Test << ".txt";

	std::ifstream ifs(ss.str());
	std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

	for (int i = 0; 6 * i + 5 < str.length(); i++){
		Indexes.push_back(str.substr(6 * i, 6));
		str.erase(str.begin() + 6 * i + 6);
	}

}

void OpenTest::OpenImages(string Directory_Images_Files, string Format,
	int Width_Zone, int Height_Zone,
	int LabVal_Test){

	Nb_Files = Indexes.size();

	int TotalSz_Zone = Width_Zone*Height_Zone;
	Data = Mat::zeros(Nb_Files, TotalSz_Zone, CV_32FC1);
	Labels = Mat::zeros(Nb_Files, 1, CV_32FC1);

	//--------------------------------------Training Set Transformation : 1st class-------------------------------------

	for (int i = 0; i < Nb_Files; i++)
	{
		stringstream ss;
		ss << Directory_Images_Files << "\\" << Indexes[i] << Format;
		Mat Image = cv::imread(ss.str(), 0);

		if (!Image.data) {
			cout << "Error occured while charging image " << Indexes[i] << Format << ".\n" << endl;
			cout << "Please check path or file format.\n\n" << endl;
			system("PAUSE");
		}

		cv::resize(Image, Image, Size(Width_Zone, Height_Zone));

		int ii = 0;
		for (int l = 0; l < Image.rows; l++) {
			for (int j = 0; j < Image.cols; j++) {
				Data.at<float>(i, ii) = Image.at<uchar>(l, j);
				ii++;
			}
		}
		Labels.at<float>(i, 0) = LabVal_Test;

	}
}

void OpenTest::Open(string Directory_Text_Files, string Directory_Images_Files, 
	string Format, string File_Test,
	int Width_Zone, int Height_Zone, int LabVal_Test){

	cout << "Charging testing data..." << endl;
	OpenTest::OpenText(Directory_Text_Files, File_Test);
	OpenTest::OpenImages(Directory_Images_Files, Format, Width_Zone, Height_Zone, LabVal_Test);
	cout << "Testing data charged.\n" << endl;
}