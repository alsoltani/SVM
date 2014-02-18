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
#include <cmath>

#include "svm.h"

using namespace cv;
using namespace std;

SVMModel::SVMModel(){}

void SVMModel::SVM_ConsoleVersion(OpenAll OA){

	CvParamGrid grid_C;
	CvParamGrid grid_Gamma;

	cout << "----------------------------------------" << endl;
	cout << "SVM Type Selection" << endl;
	cout << "----------------------------------------" << endl;
	cout << "Please select your SVM model :" << endl;
	cout << "- To specify custom parameters, please type 1." << endl;
	cout << "- To run an automatic research of optimal parameters, please type 2.\n" << endl;

	int Answ = GetLine_Int();
	if (Answ == 1)

	{
		CvSVMParams params;
		cout << "----------------------------------------" << endl;
		cout << "Custom SVM" << endl;
		cout << "----------------------------------------" << endl;
		cout << "***** KERNEL TYPES **** \n 0 : Linear \n 1 : Polynomial \n 2 : RBF \n 3 : Sigmoid" << "\n" << endl;

		cout << "Please enter values for the following parameters. If not applicable, enter 0." << endl;
		cout << "- Kernel type :" << endl;
		params.kernel_type = GetLine_Int();
		cout << "- C :" << endl;
		params.C = GetLine_Float();
		cout << "- Gamma (POLY / RBF / SIGMOID only) :" << endl;
		params.gamma = GetLine_Float();
		cout << "- Degree (POLY only) :" << endl;
		params.degree = GetLine_Int();

		clock_t begin = clock();

		SVMModel::SVM_Train_Custom(OA.OTr, params); //Training & printing training results
		SVMModel::SVM_Test(OA.OTr, OA.OTe); //Testing & printing testing results

		clock_t end = clock(); //Elapsed seconds for the inital training.
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		cout << "Elapsed seconds : " << elapsed_secs << endl;

	}
	else
	{

		cout << "----------------------------------------" << endl;
		cout << "Optimal SVM Parameters" << endl;
		cout << "----------------------------------------" << endl;
		cout << "***** KERNEL TYPES **** \n 0 : Linear \n 1 : Polynomial \n 2 : RBF \n 3 : Sigmoid" << "\n" << endl;
		cout << "Please enter number of iterations." << endl;
		int Iterations = GetLine_Int();

		cout << "Parameter C : Please enter exponent for min. grid limit." << endl;
		int p = GetLine_Int();
		grid_C.min_val = pow(10, p);
		cout << "---> Min. grid limit will be " << pow(10, p) << endl;
		cout << "Parameter C : Please enter exponent for max. grid limit." << endl;
		int k = GetLine_Int();
		grid_C.max_val = pow(10, k) + 1e-10;
		cout << "---> Max. grid limit will be " << pow(10, k)+1e-10 << endl;
		grid_C.step = 10;

		cout << "Parameter Gamma : Please enter exponent for min. grid limit." << endl;
		int m = GetLine_Int();
		grid_Gamma.min_val = pow(10, m);
		cout << "---> Min. grid limit will be " << pow(10,m) << endl;
		cout << "Parameter Gamma : Please enter exponent for max. grid limit." << endl;
		int n = GetLine_Int();
		grid_Gamma.max_val = pow(10, n) + 1e-10;
		cout << "---> Max. grid limit will be " << pow(10, n) + 1e-10 << endl;
		grid_Gamma.step = 10;

		for (int j = 1; j < Iterations + 1; j++)
		{
			cout << "\n"<<"--STEP " << j << "--" << endl;
			CvSVMParams params;
			clock_t begin = clock();

			SVMModel::SVM_Train_Optimal(OA.OTr, grid_C, grid_Gamma, params, j);
			SVMModel::SVM_Test(OA.OTr, OA.OTe);

			clock_t end = clock(); //Elapsed seconds for the inital training.
			double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
			cout << "Elapsed seconds : " << elapsed_secs << endl;
		}
	}
}
void SVMModel::SVM_Train_Custom(OpenTrain OTr, CvSVMParams params)
{

	svm.train(OTr.Data, OTr.Labels, Mat(), Mat(), params);
	PrintParam(params);
}

void SVMModel::SVM_Train_Optimal(OpenTrain OTr, CvParamGrid grid_C, CvParamGrid grid_Gamma, CvSVMParams params, int j){

		//---------------------------------------------------Training the model--------------------------------------------------

		svm.train_auto(OTr.Data, OTr.Labels, Mat(), Mat(), params, 2, grid_C, grid_Gamma);
		params = svm.get_params();

		grid_C.max_val = (params.C)*(pow(10, 1.0 / (2 * j - 1)) + pow(10, 1.0 / (2 * j))) / 2;
		grid_C.min_val = (params.C) / (pow(10, 1.0 / (2 * j)));
		grid_C.step = (pow(10, 1.0 / (2 * j)));
		grid_Gamma.min_val = (params.gamma) / (pow(10, 1.0 / (2 * j)));
		grid_Gamma.max_val = (params.gamma)*(pow(10, 1.0 / (2 * j - 1)) + pow(10, 1.0 / (2 * j))) / 2;
		grid_Gamma.step = (pow(10, 1.0 / (2 * j)));

		PrintParamGrid(params, grid_Gamma, grid_C);

	}

void SVMModel::SVM_Test(OpenTrain OTr, OpenTest OTe){

	Mat Results; // Empty mat used for classification output
	int CountMisclass(0); //For missclassification

	svm.predict(OTe.Data, Results);

	for (int i = 0; i < OTe.Nb_Files; i++){
		CountMisclass += std::abs(OTe.Labels.at<float>(i,0) - Results.at<float>(i, 0));
	}

	cout << OTe.Nb_Files << "elements were tested." << endl;
	cout << CountMisclass << " elements have been misclassified." << endl;
	cout << "Misclassification rate : " << (CountMisclass + 0.0) / OTe.Nb_Files << endl;

}

void SVMModel::PrintParamGrid(CvSVMParams params, CvParamGrid grid_Gamma, CvParamGrid grid_C){

	//Prints statmodel parameters and associated grids.
	cout << "Kernel Type : " << params.kernel_type << endl;
	cout << "C : " << params.C << endl;
	cout << "Gamma : " << params.gamma << endl;
	cout << "P : " << params.p << endl;
	cout << "Nu : " << params.nu << endl;
	cout << "Coef : " << params.coef0 << endl;
	cout << "Degree : " << params.degree << "\n" << endl;
	cout << "Grid - Gamma : " << grid_Gamma.min_val << "::" << grid_Gamma.max_val << endl;
	cout << "Grid - C     : " << grid_C.min_val << "::" << grid_C.max_val << "\n" << endl;
}

void SVMModel::PrintParam(CvSVMParams params){

	cout << "Kernel Type : " << params.kernel_type << endl;
	cout << "C : " << params.C << endl;
	cout << "Gamma : " << params.gamma << endl;
	cout << "P : " << params.p << endl;
	cout << "Nu : " << params.nu << endl;
	cout << "Coef : " << params.coef0 << endl;
	cout << "Degree : " << params.degree << "\n" << endl;
}