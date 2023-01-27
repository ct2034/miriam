#include <boost/program_options.hpp>
#include <boost/python.hpp>

#include <iostream>

#include "lodepng.h"
// #include <torch/torch.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
using namespace std::chrono;

int main(int argc, char ** argv)
{
  // Options
  std::string mapFile;
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
    "pngfile,p",
    po::value<std::string>(&mapFile)->required(),
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
//   mapFile = argv[1];

  float DA = 0.14;
  float DB = 0.06;
  float f = 0.035;
  float k = 0.065;
  float delta_t = 1.0;
  int iterations = 10000;

  // boost::python::list l = run(mapFile, DA, DB, f, k, delta_t, iterations);
  // if (l == boost::python::list()) {
  //   return 1;
  // }
  return 0;
}

class Gsorm
{
public:
  Gsorm()
  {
  }

  boost::python::tuple run(
    std::string mapFile, float DA, float DB, float f, float k, float delta_t,
    int iterations, int resolution)
  {
    // Load PNG image
    std::vector<unsigned char> image; // the raw pixels
    unsigned width, height;

    // decode
    unsigned error = lodepng::decode(image, width, height, mapFile, LCT_GREY, 8);
    if (error) {
      std::cerr << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
      return {};
    }
    std::cout << "Loaded png: " << width << "x" << height << std::endl;

//   torch::Tensor tensor = torch::rand({5, 5});
//   std::cout << tensor << std::endl;

    cv::Mat image_in = cv::imread(mapFile, cv::IMREAD_GRAYSCALE);
    cv::Mat mask;
    cv::inRange(image_in, cv::Scalar(0), cv::Scalar(254), mask);
    cv::resize(mask, mask, cv::Size(resolution, resolution));
    // cv::imwrite("mask.png", mask);
    std::cout << "Loaded mask: " << mask.size() << std::endl;

    // grey scott model
    cv::Mat a = cv::Mat::zeros(mask.size(), CV_32F);
    cv::Mat b = cv::Mat::zeros(mask.size(), CV_32F);
    cv::Mat a_diff = cv::Mat::zeros(mask.size(), CV_32F);
    cv::Mat b_diff = cv::Mat::zeros(mask.size(), CV_32F);
    cv::Mat lap_a = cv::Mat::zeros(mask.size(), CV_32F);
    cv::Mat lap_b = cv::Mat::zeros(mask.size(), CV_32F);

    // initialize a and b
    // TODO: seed
    cv::randu(a, cv::Scalar(0.8), cv::Scalar(1.0));
    cv::randu(b, cv::Scalar(0.0), cv::Scalar(0.2));

    auto start = high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
      // set A and B to 0 where mask is 255
      a.setTo(0, mask);
      b.setTo(0, mask);

      // save image for debugging
      // if (i % 1000 == 0) {
      //   std::cout << "Iteration: " << i << std::endl;
      //   cv::imwrite("a_" + std::to_string(i) + ".png", a * 255);
      // }

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
    cv::Mat spots_layer = cv::Mat::zeros(mask.size(), CV_8U);
    spots_layer.setTo(255, b > 0.5 * b_max);
//   cv::imwrite("spots_layer.png", spots_layer);

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

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    float duration_ms = static_cast<float>(duration.count()) / 1000.;
    std::cout << "Time taken by function: " <<
      duration_ms << " ms" << std::endl;

    // draw middles
    // cv::Mat middles_layer = cv::Mat::zeros(mask.size(), CV_8U);
    // for (auto middle : middles) {
    //   cv::circle(middles_layer, middle, 1, cv::Scalar(255), -1);
    // }
    // cv::imwrite("middles_layer.png", middles_layer);

    // overlay middles on spots
    // cv::Mat spots_middles_layer = cv::Mat::zeros(mask.size(), CV_8UC3);
    // cv::cvtColor(spots_layer, spots_middles_layer, cv::COLOR_GRAY2BGR);
    // cv::cvtColor(middles_layer, middles_layer, cv::COLOR_GRAY2BGR);
    // cv::addWeighted(spots_middles_layer, 0.3, middles_layer, 0.7, 0, spots_middles_layer);
    // cv::imwrite("spots_middles_layer.png", spots_middles_layer);

    boost::python::list points;
    for (auto middle : middles) {
      boost::python::tuple point = boost::python::make_tuple(middle.x, middle.y);
      points.append(point);
    }
    return boost::python::make_tuple(points, duration_ms);
  }
};

BOOST_PYTHON_MODULE(libgsorm)
{
  using namespace boost::python;
  namespace bp = boost::python;
  class_<Gsorm>("Gsorm", init<>())
  .def(
    "run", &Gsorm::run,
    (bp::arg("mapFile"), bp::arg("DA"), bp::arg("DB"), bp::arg("f"),
    bp::arg("k"), bp::arg("delta_t"), bp::arg("iterations"), bp::arg("resolution")))
  ;
}
