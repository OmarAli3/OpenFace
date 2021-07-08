///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////


// FeatureExtraction.cpp : Defines the entry point for the feature extraction console application.

// Local includes
#include "LandmarkCoreIncludes.h"

#include <FaceAnalyser.h>
#include <SequenceCapture.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h>

#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif
#include <ImageManipulationHelpers.h>

namespace py = pybind11;

cv::Mat numpy_to_cv_mat(py::array_t<unsigned char>& input) {

	if (input.ndim() != 3)
		throw std::runtime_error("3-channel image must be 3 dims ");

	py::buffer_info buf = input.request();

	cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

	return mat;
}

class AUs {
public:
	AUs(const std::string& models_path) {
		arguments.push_back(models_path);
		arguments.push_back("-aus");
		// Load face landmark detector
		det_parameters = new LandmarkDetector::FaceModelParameters(arguments);
		face_model = new LandmarkDetector::CLNF((*det_parameters).model_location);

		if (!(*face_model).loaded_successfully)
		{
			throw std::runtime_error("ERROR: Could not load the landmark detector");
		}

		// Load facial feature extractor and AU analyser
		face_analysis_params = new FaceAnalysis::FaceAnalyserParameters(arguments);
		face_analyser = new FaceAnalysis::FaceAnalyser(*face_analysis_params);

		if ((*face_analyser).GetAUClassNames().size() == 0 && (*face_analyser).GetAUClassNames().size() == 0)
		{
			throw std::runtime_error("WARNING: no Action Unit models found");
		}

	}

	void add_frame(py::array_t<unsigned char>& input) {
		frame = numpy_to_cv_mat(input);
		Utilities::ConvertToGrayscale_8bit(frame, gray_frame);
	}
	void predict_aus() {
		if (!frame.empty())
		{
			// The actual facial landmark detection / tracking
			bool detection_success = LandmarkDetector::DetectLandmarksInVideo(frame, *face_model, *det_parameters, gray_frame);

			// Do face alignment
			cv::Mat sim_warped_img;
			cv::Mat_<double> hog_descriptor; int num_hog_rows = 0, num_hog_cols = 0;

			// Perform AU detection and HOG feature extraction, as this can be expensive only compute it if needed by output or visualization
			(*face_analyser).AddNextFrame(frame, (*face_model).detected_landmarks, (*face_model).detection_success, 0, false);
			(*face_analyser).GetLatestAlignedFace(sim_warped_img);
			(*face_analyser).GetLatestHOG(hog_descriptor, num_hog_rows, num_hog_cols);

			std::vector<std::pair<std::string, std::vector<double>>> aus = (*face_analyser).PostprocessOutputFile();
			for (auto& [a, b] : aus) aus_dict[a.c_str()] = b[0];

		}
	}
	py::dict get_last_predictions() {
		(*face_analyser).Reset();
		(*face_model).Reset();
		return aus_dict;
	}

	py::dict predict(py::array_t<unsigned char>& input) {
		add_frame(input);
		predict_aus();
		return get_last_predictions();

	}
	~AUs() {
		delete det_parameters;
		delete face_model;
		delete face_analysis_params;
		delete face_analyser;
	}

private:
	std::vector<std::string> arguments;
	LandmarkDetector::FaceModelParameters* det_parameters;
	LandmarkDetector::CLNF* face_model;
	FaceAnalysis::FaceAnalyserParameters* face_analysis_params;
	FaceAnalysis::FaceAnalyser* face_analyser;
	cv::Mat frame;
	cv::Mat_<uchar> gray_frame;
	py::dict aus_dict;
};

void add() {
	std::cout << "hello" << std::endl;
}
PYBIND11_MODULE(openface2, m) {
	m.doc() = "python binding for openFace2";

	py::class_<AUs>(m, "AUs")
		.def(py::init<const std::string&>())
		.def("add_frame", &AUs::add_frame)
		.def("predict_aus", &AUs::predict_aus)
		.def("get_last_predictions", &AUs::get_last_predictions)
		.def("predict", &AUs::predict);

}
