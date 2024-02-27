#include <boost/program_options.hpp>
#include <boost/python.hpp>

#include <iostream>

#include "lodepng.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
using namespace std::chrono;

class CVT
{
public:
  CVT()
  {
  }

  boost::python::tuple run(
    std::string mapFile, int n_nodes, int seed = 1)
  {
    std::string example_folder = "roadmaps/cvt/examples/";

    // seed opencv rng
    cv::RNG rng(seed);

    // Load PNG image
    std::vector<unsigned char> image; // the raw pixels
    unsigned width, height;
    unsigned error = lodepng::decode(image, width, height, mapFile, LCT_GREY, 8);
    if (error) {
      std::cerr << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
      return {};
    }
    std::cout << "Loaded png: " << width << "x" << height << std::endl;

    // Initialize Points
    auto start = high_resolution_clock::now();
    std::vector<cv::Point> nodes;
    while (nodes.size() < n_nodes) {
      float x = rng.uniform((float) 0, (float) width);
      float y = rng.uniform((float) 0, (float) height);
      if (image[(int) y * width + (int) x] == 255) {
        nodes.push_back(cv::Point(x, y));
      }
    }

    // Make Voronoi diagram
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8U);
    for (int i = 0; i < mask.rows; i++) {
      for (int j = 0; j < mask.cols; j++) {
        if (image[i * width + j] == 255) {
          mask.at<uchar>(i, j) = 255;
        }
      }
    }
    cv::Subdiv2D subdiv(cv::Rect(50, 50, 910, 910));
    for (auto node : nodes) {
      subdiv.insert(node);
    }

    // Draw Voronoi diagram
    cv::Mat img_voronoi = cv::Mat::zeros(mask.size(), CV_8UC3);
    std::vector<std::vector<cv::Point2f>> facets;
    std::vector<cv::Point2f> centers;
    subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);
    for (size_t i = 0; i < facets.size(); i++) {
      std::vector<cv::Point> facet;
      for (size_t j = 0; j < facets[i].size(); j++) {
        facet.push_back(facets[i][j]);
      }
      cv::fillConvexPoly(img_voronoi, facet, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 8, 0);
    }
    cv::imwrite(example_folder + "voronoi.png", img_voronoi);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    float duration_total_ms = static_cast<float>(duration.count()) / 1000.;
    std::cout << "Total time taken by function: " <<
      duration_total_ms << " ms" << std::endl;

    // if (plot) {
    //   // draw middles
    //   cv::Mat middles_layer = cv::Mat::zeros(mask.size(), CV_8U);
    //   for (auto middle : middles) {
    //     cv::circle(middles_layer, middle, 1, cv::Scalar(255), -1);
    //   }

    //   // overlay middles on spots
    //   cv::Mat spots_middles_layer = cv::Mat::zeros(mask.size(), CV_8UC3);
    //   cv::cvtColor(spots_layer, spots_middles_layer, cv::COLOR_GRAY2BGR);
    //   cv::cvtColor(middles_layer, middles_layer, cv::COLOR_GRAY2BGR);
    //   cv::addWeighted(spots_middles_layer, 0.3, middles_layer, 0.7, 0, spots_middles_layer);
    //   cv::imwrite(example_folder + "spots_middles_layer.png", spots_middles_layer);
    // }

    boost::python::list points;
    for (auto node : nodes) {
      points.append(boost::python::make_tuple(node.x, node.y));
    }
    return boost::python::make_tuple(points, duration_total_ms);
  }
};

BOOST_PYTHON_MODULE(libcvt)
{
  using namespace boost::python;
  namespace bp = boost::python;
  class_<CVT>("CVT", init<>())
  .def(
    "run", &CVT::run,
    (bp::arg("mapFile"), bp::arg("n_nodes"), bp::arg("seed") = 0)
  )
  ;
}
