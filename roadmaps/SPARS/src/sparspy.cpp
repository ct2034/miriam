#include <boost/python.hpp>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

// Boost
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>

// Eigen
// #include <Eigen/Core>

// Yaml
#include <yaml-cpp/yaml.h>

// OMPL
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/prm/SPARS.h>
#include <ompl/geometric/planners/prm/SPARStwo.h>
#include <ompl/base/terminationconditions/IterationTerminationCondition.h>

// LodePNG
#include "lodepng.h"

// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace ompl;
using namespace std::chrono;

struct GenerationType
{
  enum Type
  {
    SPARS,
    SPARS2,
  };
};

std::istream & operator>>(std::istream & in, GenerationType::Type & type)
{
  std::string token;
  in >> token;
  if (token == "SPARS") {
    type = GenerationType::SPARS;
  } else if (token == "SPARS2") {
    type = GenerationType::SPARS2;
  } else {
    in.setstate(std::ios_base::failbit);
  }
  return in;
}

class ImageStateValidityChecker : public ompl::base::StateValidityChecker
{
public:
  ImageStateValidityChecker(
    ompl::base::SpaceInformationPtr si,
    const std::vector<unsigned char> & image,
    unsigned width,
    unsigned height)
  : StateValidityChecker(si), image_(image), width_(width), height_(height)
  {
  }

  bool isValid(const ompl::base::State * state) const
  {
    const ompl::base::RealVectorStateSpace::StateType * typedState =
      state->as<ompl::base::RealVectorStateSpace::StateType>();

    int x = (int)(*typedState)[0];
    int y = (int)(*typedState)[1];

    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
      return false;
    }

    return image_[y * width_ + x] >= 255;
  }

private:
  const std::vector<unsigned char> & image_;
  unsigned width_;
  unsigned height_;
};

// class ImageMotionValidator : public ompl::base::MotionValidator
// {
// public:
//   ImageMotionValidator(
//     ompl::base::SpaceInformationPtr si,
//     const std::vector<unsigned char> & image,
//     unsigned width,
//     unsigned height)
//   : MotionValidator(si), image_(image), width_(width), height_(height), stateValidityChecker_(si,
//       image,
//       width,
//       height)
//   {
//   }

//   bool checkMotion(const ompl::base::State * s1, const ompl::base::State * s2) const
//   {
//     std::pair<ompl::base::State *, double> lastValid;
//     return checkMotion(s1, s2, lastValid);
//   }

//   bool checkMotion(
//     const ompl::base::State * s1, const ompl::base::State * s2,
//     std::pair<ompl::base::State *, double> & lastValid) const
//   {
//     const ompl::base::RealVectorStateSpace::StateType * ts1 =
//       s1->as<ompl::base::RealVectorStateSpace::StateType>();
//     int x1 = (int)(*ts1)[0];
//     int y1 = (int)(*ts1)[1];
//     const ompl::base::RealVectorStateSpace::StateType * ts2 =
//       s2->as<ompl::base::RealVectorStateSpace::StateType>();
//     int x2 = (int)(*ts2)[0];
//     int y2 = (int)(*ts2)[1];

//     if (not (
//         stateValidityChecker_.isValid(s1) &&
//         stateValidityChecker_.isValid(s2) ))
//     {
//       return false;
//     }

//     int timesteps = std::max(std::abs(x2 - x1), std::abs(y2 - y1)) + 1;
//     for (int i = 0; i < timesteps; i++) {
//       float t = (float)i / (float)timesteps;
//       int x = (int)((1.0 - t) * x1 + t * x2);
//       int y = (int)((1.0 - t) * y1 + t * y2);
//       std::cout << "x: " << x << " y: " << y << std::endl;
//       if (image_[y * width_ + x] < 255) {
//         std::cout << "false" << std::endl;
//         return false;
//       }
//     }
//     std::cout << "true" << std::endl;
//     return true;
//   }

// private:
//   ImageStateValidityChecker stateValidityChecker_;
//   const std::vector<unsigned char> & image_;
//   unsigned width_;
//   unsigned height_;
// };

class Spars
{
public:
  Spars()
  {
  }

  boost::python::tuple run(
    std::string mapFile, int seed,
    float denseDelta, float sparseDelta, float stretchFactor,
    int maxFailures, double maxTime, int maxIter)
  {
    ompl::RNG::setSeed(seed);

    const size_t dimension = 2;

    std::vector<unsigned char> image;   // the raw pixels
    unsigned width, height;

    // decode
    unsigned error = lodepng::decode(image, width, height, mapFile, LCT_GREY, 8);

    // if there's an error, display it
    if (error) {
      std::cerr << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
      return {};
    }

    std::cout << "Loaded environment: " << width << "x" << height << std::endl;

    // YAML::Node cfg = YAML::LoadFile(configFile);

    base::StateSpacePtr space(new base::RealVectorStateSpace(dimension));
    base::RealVectorBounds bounds(dimension);
    bounds.setLow(0, 0);
    bounds.setHigh(0, width);
    bounds.setLow(1, 0);
    bounds.setHigh(1, height);
    space->as<base::RealVectorStateSpace>()->setBounds(bounds);

    base::SpaceInformationPtr si(new base::SpaceInformation(space));
    base::StateValidityCheckerPtr stateValidityChecker(
      new ImageStateValidityChecker(
        si, image,
        width, height));
    si->setStateValidityChecker(stateValidityChecker);
    si->setStateValidityCheckingResolution(1. / (float)std::max(width, height) / 5.);
    // base::MotionValidatorPtr motionValidator(
    //   new ImageMotionValidator(
    //     si, image,
    //     width, height));
    // si->setMotionValidator(
    //   ompl::base::DiscreteMotionValidatorPtr(
    //     new ompl::base::DiscreteMotionValidator(si)));
    si->setup();

    base::ProblemDefinitionPtr pdef(new base::ProblemDefinition(si));
    base::ScopedState<base::RealVectorStateSpace> start(space);
    // fill start state
    base::ScopedState<base::RealVectorStateSpace> goal(space);
    // fill goal state
    // pdef->setStartAndGoalStates(start, goal);  // TODO: why this?

    geometric::SPARStwo p(si);
    p.setProblemDefinition(pdef);
    // std::cout << p.getDenseDeltaFraction()
    // << "Delta: " << p.getSparseDeltaFraction() * si->getMaximumExtent()
    // << " " << p.getStretchFactor()
    // << " " << p.getMaxFailures() << std::endl;
    // p.setSparseDeltaFraction(1.0 / si->getMaximumExtent() );

    p.setDenseDeltaFraction(denseDelta / si->getMaximumExtent());
    p.setSparseDeltaFraction(sparseDelta / si->getMaximumExtent());
    p.setStretchFactor(stretchFactor);
    p.setMaxFailures(maxFailures);

    auto t_start = high_resolution_clock::now();
    p.constructRoadmap(
      base::IterationTerminationCondition(maxIter));
    auto t_stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(t_stop - t_start);
    float duration_ms = static_cast<float>(duration.count()) / 1000.;

    p.printDebug();

    const auto & roadmapOMPL = p.getRoadmap();
    std::cout << "#vertices " << boost::num_vertices(roadmapOMPL) << std::endl;
    std::cout << "#edges " << boost::num_edges(roadmapOMPL) << std::endl;
    // // const auto& roadmapOMPL = p.getDenseGraph();
    auto stateProperty = boost::get(geometric::SPARStwo::vertex_state_t(), roadmapOMPL);

    // save to image
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    BOOST_FOREACH(auto e, boost::edges(roadmapOMPL))
    {
      size_t i = boost::source(e, roadmapOMPL);
      size_t j = boost::target(e, roadmapOMPL);

      base::State * state_i = stateProperty[i];
      base::State * state_j = stateProperty[j];

      const base::RealVectorStateSpace::StateType * typedState_i =
        state_i->as<base::RealVectorStateSpace::StateType>();
      const base::RealVectorStateSpace::StateType * typedState_j =
        state_j->as<base::RealVectorStateSpace::StateType>();

      int x1 = (*typedState_i)[0];
      int y1 = (*typedState_i)[1];
      int x2 = (*typedState_j)[0];
      int y2 = (*typedState_j)[1];

      cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 255), 1);
    }
    cv::imwrite("spars.png", img);

    // for (size_t i = 0; i < boost::num_vertices(roadmapOMPL); ++i) {
    //   base::State* state = stateProperty[i];
    //   if (state) {
    //     const base::RealVectorStateSpace::StateType* typedState =
    //     state->as<base::RealVectorStateSpace::StateType>(); float x = (*typedState)[0]; float y =
    //     (*typedState)[1]; std::cout << "v" << i << "," << x << "," << y << std::endl;
    //   }
    // }

    // output
    boost::python::list edges;

    BOOST_FOREACH(const geometric::SPARS::SparseEdge e, boost::edges(roadmapOMPL))
    {
      size_t i = boost::source(e, roadmapOMPL);
      size_t j = boost::target(e, roadmapOMPL);

      base::State * state_i = stateProperty[i];
      base::State * state_j = stateProperty[j];
      if (state_i && state_j) {
        const auto typedState_i = state_i->as<base::RealVectorStateSpace::StateType>();
        float x_i = (*typedState_i)[0];
        float y_i = (*typedState_i)[1];

        const auto typedState_j = state_j->as<base::RealVectorStateSpace::StateType>();
        float x_j = (*typedState_j)[0];
        float y_j = (*typedState_j)[1];

        edges.append(boost::python::make_tuple(i, x_i, y_i, j, x_j, y_j));
      }
    }

    return boost::python::make_tuple(edges, duration_ms);

    // if (genType == GenerationType::SPARS2) {
    //   auto node = cfg["spars2"];

    //   geometric::SPARStwo p(si);
    //   p.setProblemDefinition(pdef);

    //   p.setDenseDeltaFraction(node["denseDelta"].as<float>() / si->getMaximumExtent() );
    //   p.setSparseDeltaFraction(node["sparseDelta"].as<float>() / si->getMaximumExtent() );
    //   p.setStretchFactor(node["stretchFactor"].as<float>());
    //   p.setMaxFailures(node["maxFailures"].as<int>());

    //   p.constructRoadmap(
    //     /*base::timedPlannerTerminationCondition(30)*/ base::IterationTerminationCondition(
    //       node["maxIter"].as<int>()), true);

    //   const auto & roadmapOMPL = p.getRoadmap();
    //   auto stateProperty = boost::get(geometric::SPARStwo::vertex_state_t(), roadmapOMPL);

    //   std::unordered_map<geometric::SPARS::SparseVertex, vertex_t> vertexMap;
    //   for (size_t i = 0; i < boost::num_vertices(roadmapOMPL); ++i) {
    //     base::State * state = stateProperty[i];
    //     if (state) {
    //       position_t p(-1, -1, -1);
    //       const base::RealVectorStateSpace::StateType * typedState =
    //         state->as<base::RealVectorStateSpace::StateType>();
    //       if (dimension == 3) {
    //         p = position_t((*typedState)[0], (*typedState)[1], (*typedState)[2]);
    //       } else if (dimension == 2) {
    //         p = position_t((*typedState)[0], (*typedState)[1], fixedZ);
    //       }
    //       auto v = boost::add_vertex(roadmap);
    //       roadmap[v].pos = p;
    //       roadmap[v].name = "v" + std::to_string(v);
    //       vertexMap[i] = v;
    //     }
    //   }

    //   // add OMPL edges
    //   BOOST_FOREACH(const geometric::SPARStwo::Edge e, boost::edges(roadmapOMPL)) {
    //     size_t i = boost::source(e, roadmapOMPL);
    //     size_t j = boost::target(e, roadmapOMPL);

    //     add_edge(vertexMap.at(i), vertexMap.at(j), roadmap);
    //   }
    // }

#if 0
    // assign edge names
    size_t i = 0;
    BOOST_FOREACH(const edge_t e, boost::edges(roadmap)) {
      roadmap[e].name = "e" + std::to_string(i);
      ++i;
    }

    saveSearchGraph(roadmap, outputFile);
#endif
    return {};
  }
};

BOOST_PYTHON_MODULE(libsparspy)
{
  using namespace boost::python;
  namespace bp = boost::python;
  class_<Spars>("Spars", init<>())
  .def(
    "run", &Spars::run,
    (bp::arg("mapFile"), bp::arg("seed"), bp::arg("denseDelta"), bp::arg("sparseDelta"),
    bp::arg("stretchFactor"), bp::arg("maxFailures"), bp::arg("maxTime"), bp::arg("maxIter"))
  );
}
