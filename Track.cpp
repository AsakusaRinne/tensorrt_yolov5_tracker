// TrackCounter.cpp: 定义应用程序的入口点。
//
#include <iostream>
#include <fstream>
#include <chrono>
#include "sort/include/tracker.h"
#include "yolov5/Loader.h"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unistd.h>
#include <vector>
#include "time.h"
using namespace std;

#define INPUT_H Yolo::INPUT_H
#define INPUT_W Yolo::INPUT_W
#define ORIGINAL_W Yolo::ORIGINAL_W
#define ORIGINAL_H Yolo::ORIGINAL_H

static string engineFile;

struct sharedData {
	std::queue<float*>* dataQueue;
	std::queue<std::vector<std::vector<Yolo::Detection>>>* detQueue;
	std::queue<cv::Mat>* imgQueue;
};

cv::Rect get_rect(int w, int h, float bbox[4]) {
	int l, r, t, b;
	float r_w = INPUT_W / (w * 1.0);
	float r_h = INPUT_H / (h * 1.0);
	if (r_h > r_w) {
		l = bbox[0] - bbox[2] / 2.f;
		r = bbox[0] + bbox[2] / 2.f;
		t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * h) / 2;
		b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * h) / 2;
		l = l / r_w;
		r = r / r_w;
		t = t / r_w;
		b = b / r_w;
	}
	else {
		l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * w) / 2;
		r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * w) / 2;
		t = bbox[1] - bbox[3] / 2.f;
		b = bbox[1] + bbox[3] / 2.f;
		l = l / r_h;
		r = r / r_h;
		t = t / r_h;
		b = b / r_h;
	}
	return cv::Rect(l, t, r - l, b - t);
}


int main(int argc, char** argv) {
	if (argc == 6 && std::string(argv[1]) == "-s") {
		yolov5_build_engine(argv[2],argv[3],atof(argv[4]),atof(argv[5]));
	}
	else if (argc == 5 && std::string(argv[1]) == "-s" && std::string(argv[1]) == "s") {
		yolov5_build_engine(argv[2], argv[3], 0.33, 0.50);
	}
	else if (argc == 5 && std::string(argv[1]) == "-s" && std::string(argv[1]) == "m") {
		yolov5_build_engine(argv[2], argv[3], 0.67, 0.75);
	}
	else if (argc == 5 && std::string(argv[1]) == "-s" && std::string(argv[1]) == "l") {
		yolov5_build_engine(argv[2], argv[3], 1.0, 1.0);
	}
	else if (argc == 5 && std::string(argv[1]) == "-s" && std::string(argv[1]) == "x") {
		yolov5_build_engine(argv[2], argv[3], 1.25, 1.33);
	}
	else if (argc == 4 && std::string(argv[1]) == "-v") {
	//initialize capture, writer, detector, and tracker
	std::string filename = argv[2];
	cv::VideoCapture cap;
	cap.open(filename);
	if (!cap.isOpened()) {
		std::cout << "failed to open video " << filename << std::endl;
		return -1;
	}
	int frameCount = cap.get(cv::CAP_PROP_FRAME_COUNT);
	int maxIndex = int(frameCount / BATCH_SIZE);

	yolov5_detector detector(argv[3]);
	Tracker sort;
	std::ofstream resultFile("testSingleThread.txt", std::ios::out);
	std::map<int, Track> tracks;
	cv::VideoWriter writer;

	auto code = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
	auto fps = cap.get(cv::CAP_PROP_FPS);
	auto size = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	writer.open("../output/result.avi", code, fps, size, true);

	//get a batch of images and preprocess them
	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
	for (int f = 0; f < maxIndex; f++) {
		auto start = std::chrono::system_clock::now();
		vector<cv::Mat> imgs;
		for (int b = 0; b < BATCH_SIZE; b++) {
			cv::Mat img;
			cap.read(img);
			//cv::Mat copy_img;
			//img.copyTo(copy_img);
			auto pr_img = preprocess_img(img);
			imgs.push_back(img);
			int i = 0;
			for (int row = 0; row < INPUT_H; ++row) {
				uchar* uc_pixel = pr_img.data + row * pr_img.step;
				for (int col = 0; col < INPUT_W; ++col) {
					data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
					data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
					data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
					uc_pixel += 3;
					++i;
				}
			}
		}

		//detect
		auto end = std::chrono::system_clock::now();
		std::cout << "Preprocess Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
		start = std::chrono::system_clock::now();
		std::vector<std::vector<Yolo::Detection>> results = detector.detect(data);
		end = std::chrono::system_clock::now();
		std::cout << "Detect Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
		start = std::chrono::system_clock::now();

		//track
		for (int b = 0; b < BATCH_SIZE; b++) {
			auto& result = results[b];
			std::vector<cv::Rect> dets;
			for (size_t j = 0; j < result.size(); j++) {
				dets.push_back(get_rect(ORIGINAL_W, ORIGINAL_H, result[j].bbox));
			}
			sort.Run(dets);
			std::map<int, Track> tracks = sort.GetTracks();

			auto im = imgs[b];
			for (auto& trk : tracks) {
				const auto& bbox = trk.second.GetStateAsBbox();
				/*Note that we will not export coasted tracks
				If we export coasted tracks, the total number of false negative will decrease (and maybe ID switch)
				However, the total number of false positive will increase more (from experiments),
				which leads to MOTA decrease
				Developer can export coasted cycles if false negative tracks is critical in the system*/
				if (trk.second.coast_cycles_ < kMaxCoastCycles
					&& (trk.second.hit_streak_ >= kMinHits || f * BATCH_SIZE + b < kMinHits)) {
					// Print to terminal for debugging

					resultFile << f * BATCH_SIZE + b << "," << trk.first << "," << bbox.tl().x << "," << bbox.tl().y
						<< "," << bbox.width << "," << bbox.height << ",1,-1,-1,-1\n";
					cv::rectangle(im, cv::Rect(bbox.tl().x, bbox.tl().y, bbox.width, bbox.height), cv::Scalar(0x27, 0xC1, 0x36), 2);
					cv::putText(im, std::to_string(trk.first), cv::Point(bbox.tl().x, bbox.tl().y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
				}
			}
			writer.write(im);
		}
		end = std::chrono::system_clock::now();
		std::cout << "Track Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
		imgs.clear();
	}
	detector.release();
	resultFile.close();
	cap.release();
	writer.release();
	}
	return 0;
}
