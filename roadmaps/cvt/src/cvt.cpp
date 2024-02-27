#include <boost/program_options.hpp>
#include <boost/python.hpp>

#include <iostream>

#include "lodepng.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
using namespace std::chrono;

// includes for defining the Voronoi diagram adaptor
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>
// typedefs for defining the adaptor
typedef CGAL::Exact_predicates_inexact_constructions_kernel                  K;
typedef CGAL::Delaunay_triangulation_2<K>                                    DT;
typedef CGAL::Delaunay_triangulation_adaptation_traits_2<DT>                 AT;
typedef CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<DT> AP;
typedef CGAL::Voronoi_diagram_2<DT,AT,AP>                                    VD;
// typedef for the result type of the point location
typedef AT::Site_2                    Site_2;
typedef AT::Point_2                   Point_2;
typedef VD::Locate_result             Locate_result;
typedef VD::Vertex_handle             Vertex_handle;
typedef VD::Face_handle               Face_handle;
typedef VD::Halfedge_handle           Halfedge_handle;
typedef VD::Ccb_halfedge_circulator   Ccb_halfedge_circulator;


class CVT
{
public:
  CVT()
  {
    std::cout << "init" << std::endl;
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
    VD vd;
    for(auto node : nodes) {
      Site_2  site = Point_2(node.x, node.y);
      vd.insert(site);
    }
    bool valid = vd.is_valid();
    if (!valid) {
      std::cerr << "Voronoi diagram is not valid" << std::endl;
      return {};
    } else {
      std::cout << "Voronoi diagram is valid" << std::endl;
    }

    std::cout << "Number of faces: " << vd.number_of_faces() << std::endl;

    // Draw Voronoi tessellation
    cv::Mat img_voronoi = cv::Mat::zeros(height, width, CV_8UC3);
    for (auto face = vd.faces_begin(); face != vd.faces_end(); ++face) {
      Ccb_halfedge_circulator ec_start = face->ccb();
      Ccb_halfedge_circulator ec = ec_start;
      std::vector<cv::Point> face_points;
      do {
        if (!ec->has_source() or !ec->has_target()) {
          continue;
        }
        face_points.push_back(cv::Point(ec->source()->point().x(), ec->source()->point().y()));
        face_points.push_back(cv::Point(ec->target()->point().x(), ec->target()->point().y()));
      } while (++ec != ec_start);
      bool is_finite = true;
      for (auto point : face_points) {
        if (point.x < 0 or point.x >= width or point.y < 0 or point.y >= height) {
          is_finite = false;
          break;
        }
      }
      // if (is_finite) {
        // random color
        cv::Scalar color(
          rng.uniform(200, 255), 
          rng.uniform(200, 255), 
          rng.uniform(200, 255));
        cv::fillConvexPoly(img_voronoi, face_points, color);
      // }
    }

    // Mask freespace
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (image[y * width + x] != 255) {
          img_voronoi.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
        }
      }
    }

    // Also draw points
    for (auto node : nodes) {
      cv::circle(img_voronoi, node, 3, cv::Scalar(0, 0, 255), -1);
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
