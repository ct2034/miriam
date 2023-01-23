#include <boost/program_options.hpp>

#include <iostream>

#include "lodepng.h"
// #include <torch/torch.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


int main(int argc, char ** argv)
{
  // Options
  std::string pngFile;
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
    "pngfile,p",
    po::value<std::string>(&pngFile)->required(),
    "input file for environment (PNG)"
  );
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }
  } catch (po::error & e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }
//   pngFile = argv[1];

  // Load PNG image
  std::vector<unsigned char> image;  // the raw pixels
  unsigned width, height;

  // decode
  unsigned error = lodepng::decode(image, width, height, pngFile, LCT_GREY, 8);
  if (error) {
    std::cerr << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    return 1;
  }
  std::cout << "Loaded png: " << width << "x" << height << std::endl;

//   torch::Tensor tensor = torch::rand({5, 5});
//   std::cout << tensor << std::endl;

  cv::Mat image_in = cv::imread(pngFile, cv::IMREAD_GRAYSCALE);
  std::cout << "image max: " << (int) *std::max_element(
    image_in.begin<unsigned char>(),
    image_in.end<unsigned char>()) << std::endl;
  std::cout << "image min: " << (int) *std::min_element(
    image_in.begin<unsigned char>(),
    image_in.end<unsigned char>()) << std::endl;

  // mask where image is not white
  cv::Mat mask;
  cv::inRange(image_in, cv::Scalar(0), cv::Scalar(254), mask);
  // resize mask
  cv::resize(mask, mask, cv::Size(200, 200));
  std::cout << "mask max: " << (int) *std::max_element(
    mask.begin<unsigned char>(),
    mask.end<unsigned char>()) << std::endl;
  std::cout << "mask min: " << (int) *std::min_element(
    mask.begin<unsigned char>(),
    mask.end<unsigned char>()) << std::endl;
  cv::imwrite("mask.png", mask);

  // grey scott model
  cv::Mat a = cv::Mat::zeros(mask.size(), CV_32F);
  cv::Mat b = cv::Mat::zeros(mask.size(), CV_32F);
  cv::Mat a_diff = cv::Mat::zeros(mask.size(), CV_32F);
  cv::Mat b_diff = cv::Mat::zeros(mask.size(), CV_32F);
  cv::Mat lap_a = cv::Mat::zeros(mask.size(), CV_32F);
  cv::Mat lap_b = cv::Mat::zeros(mask.size(), CV_32F);

  // initialize a and b
  cv::randu(a, cv::Scalar(0.8), cv::Scalar(1.0));
  cv::randu(b, cv::Scalar(0.0), cv::Scalar(0.2));

//   // set regions to 1
//   cv::Mat roi = a(cv::Rect(400, 400, 300, 300));
//   roi.setTo(0.5);
//   roi = b(cv::Rect(400, 400, 300, 300));
//   roi.setTo(0.25);

  float DA = 0.14;
  float DB = 0.06;
  float f = 0.035;
  float k = 0.065;
  float delta_t = 1.0;

  for (int i = 0; i < 10000; i++) {
    // set A and B to 0 where mask is 255
    a.setTo(0, mask);
    b.setTo(0, mask);

    // save image for debugging
    if (i % 1000 == 0) {
      std::cout << "Iteration: " << i << std::endl;
      cv::imwrite("a_" + std::to_string(i) + ".png", a * 255);
    }

    // laplacian
    cv::Laplacian(a, lap_a, CV_32F);
    cv::Laplacian(b, lap_b, CV_32F);

    // update a and b
    // diff_A = (DA*LA - A*B**2 + f*(1-A)) * delta_t
    // diff_B = (DB*LB + A*B**2 - (k+f)*B) * delta_t
    a_diff = (DA * lap_a - a.mul(b.mul(b)) + f * (1 - a) ) * delta_t;
    b_diff = (DB * lap_b + a.mul(b.mul(b)) - (k + f) * b) * delta_t;

    a = a + a_diff;
    b = b + b_diff;
  }

  float b_max = *std::max_element(
    b.begin<float>(),
    b.end<float>());
  std::cout << "b max: " << b_max << std::endl;
  std::cout << "b min: " << *std::min_element(
    b.begin<float>(),
    b.end<float>()) << std::endl;

  cv::Mat spots_layer = cv::Mat::zeros(mask.size(), CV_8U);
  spots_layer.setTo(255, b > 0.5 * b_max);
  cv::imwrite("spots_layer.png", spots_layer);

  // find spots
  std::vector<std::vector<cv::Point>> spots;
  cv::findContours(spots_layer, spots, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  std::cout << "Found " << spots.size() << " spots" << std::endl;

  // find middles of spots
  std::vector<cv::Point> middles;
  for (auto spot : spots) {
    cv::Moments m = cv::moments(spot);
    middles.push_back(cv::Point(m.m10 / m.m00, m.m01 / m.m00));
  }

  // draw middles
  cv::Mat middles_layer = cv::Mat::zeros(mask.size(), CV_8U);
  for (auto middle : middles) {
    cv::circle(middles_layer, middle, 1, cv::Scalar(255), -1);
  }
  cv::imwrite("middles_layer.png", middles_layer);

  // overlay middles on spots
  cv::Mat spots_middles_layer = cv::Mat::zeros(mask.size(), CV_8UC3);
  cv::cvtColor(spots_layer, spots_middles_layer, cv::COLOR_GRAY2BGR);
  cv::cvtColor(middles_layer, middles_layer, cv::COLOR_GRAY2BGR);
  cv::addWeighted(spots_middles_layer, 0.3, middles_layer, 0.7, 0, spots_middles_layer);
  cv::imwrite("spots_middles_layer.png", spots_middles_layer);

  return 0;
}
