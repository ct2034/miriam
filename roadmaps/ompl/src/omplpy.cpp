#include <boost/python.hpp>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

// Boost
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>

// OMPL
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/prm/SPARS.h>
#include <ompl/geometric/planners/prm/SPARStwo.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/base/terminationconditions/IterationTerminationCondition.h>

// LodePNG
#include "lodepng.h"

// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace ompl;
using namespace std::chrono;

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

class Ompl
{
public:
  Ompl()
  {
  }

  boost::python::tuple runSparsTwo(
    std::string mapFile, int seed,
    float denseDelta, float sparseDelta, float stretchFactor,
    int maxFailures, double maxTime, int maxIter)
  {
    ompl::RNG::setSeed(seed);

    const size_t dimension = 2;

    std::vector<unsigned char> image;   // the raw pixels
    unsigned width, height;

    // decode
    unsigned error =
      lodepng::decode(image, width, height, mapFile, LCT_GREY, 8);

    // if there's an error, display it
    if (error) {
      std::cerr << "decoder error " << error << ": " <<
        lodepng_error_text(error) << std::endl;
      return {};
    }

    std::cout << "Loaded environment: " << width << "x" << height << std::endl;
    // for (
    //   std::vector<unsigned char>::const_iterator i = image.begin();
    //   i != image.end(); ++i)
    // {
    //   std::cout << *i << ' ';
    // }

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
    si->setStateValidityCheckingResolution(
      1. / (float)std::max(
        width,
        height) / 5.);
    si->setup();

    base::ProblemDefinitionPtr pdef(new base::ProblemDefinition(si));
    base::ScopedState<base::RealVectorStateSpace> start(space);
    base::ScopedState<base::RealVectorStateSpace> goal(space);

    geometric::SPARS p(si);
    p.setProblemDefinition(pdef);

    p.setDenseDeltaFraction(denseDelta / si->getMaximumExtent());
    p.setSparseDeltaFraction(sparseDelta / si->getMaximumExtent());
    p.setStretchFactor(stretchFactor);
    p.setMaxFailures(maxFailures);

    std::cout << "a" << std::endl;
    auto t_start = high_resolution_clock::now();
    std::cout << "b" << std::endl;
    p.constructRoadmap(
      base::IterationTerminationCondition(maxIter));
    std::cout << "c" << std::endl;
    auto t_stop = high_resolution_clock::now();
    std::cout << "d" << std::endl;
    auto duration = duration_cast<microseconds>(t_stop - t_start);
    std::cout << "e" << std::endl;
    float duration_ms = static_cast<float>(duration.count()) / 1000.;
    std::cout << "f" << std::endl;

    p.printDebug();

    const auto & roadmapOMPL = p.getRoadmap();
    std::cout << "#vertices " << boost::num_vertices(roadmapOMPL) << std::endl;
    std::cout << "#edges " << boost::num_edges(roadmapOMPL) << std::endl;
    // // const auto& roadmapOMPL = p.getDenseGraph();
    auto stateProperty = boost::get(
      geometric::SPARS::vertex_state_t(), roadmapOMPL);

    // output
    boost::python::list edges;

    BOOST_FOREACH(
      const geometric::SPARS::SparseEdge e, boost::edges(
        roadmapOMPL))
    {
      size_t i = boost::source(e, roadmapOMPL);
      size_t j = boost::target(e, roadmapOMPL);

      base::State * state_i = stateProperty[i];
      base::State * state_j = stateProperty[j];
      if (state_i && state_j) {
        const auto typedState_i =
          state_i->as<base::RealVectorStateSpace::StateType>();
        float x_i = (*typedState_i)[0];
        float y_i = (*typedState_i)[1];

        const auto typedState_j =
          state_j->as<base::RealVectorStateSpace::StateType>();
        float x_j = (*typedState_j)[0];
        float y_j = (*typedState_j)[1];

        edges.append(boost::python::make_tuple(i, x_i, y_i, j, x_j, y_j));
      }
    }

    return boost::python::make_tuple(edges, duration_ms);

  }

  boost::python::tuple runPRMStar(
    std::string mapFile, int seed,
    float denseDelta, float sparseDelta, float stretchFactor,
    int maxFailures, double maxTime, int maxIter)
  {
    ompl::RNG::setSeed(seed);

    const size_t dimension = 2;

    std::vector<unsigned char> image;   // the raw pixels
    unsigned width, height;

    // decode
    unsigned error =
      lodepng::decode(image, width, height, mapFile, LCT_GREY, 8);

    // if there's an error, display it
    if (error) {
      std::cerr << "decoder error " << error << ": " <<
        lodepng_error_text(error) << std::endl;
      return {};
    }

    std::cout << "Loaded environment: " << width << "x" << height << std::endl;
    // for (
    //   std::vector<unsigned char>::const_iterator i = image.begin();
    //   i != image.end(); ++i)
    // {
    //   std::cout << *i << ' ';
    // }

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
    si->setStateValidityCheckingResolution(
      1. / (float)std::max(
        width,
        height) / 5.);
    si->setup();

    base::ProblemDefinitionPtr pdef(new base::ProblemDefinition(si));
    base::ScopedState<base::RealVectorStateSpace> start(space);
    base::ScopedState<base::RealVectorStateSpace> goal(space);

    geometric::PRMstar p(si);
    p.setProblemDefinition(pdef);

    // p.setDenseDeltaFraction(denseDelta / si->getMaximumExtent());
    // p.setSparseDeltaFraction(sparseDelta / si->getMaximumExtent());
    // p.setStretchFactor(stretchFactor);
    // p.setMaxFailures(maxFailures);

    std::cout << "a" << std::endl;
    auto t_start = high_resolution_clock::now();
    std::cout << "b" << std::endl;
    p.constructRoadmap(
      base::IterationTerminationCondition(maxIter));
    std::cout << "c" << std::endl;
    auto t_stop = high_resolution_clock::now();
    std::cout << "d" << std::endl;
    auto duration = duration_cast<microseconds>(t_stop - t_start);
    std::cout << "e" << std::endl;
    float duration_ms = static_cast<float>(duration.count()) / 1000.;
    std::cout << "f" << std::endl;

    // p.printDebug();

    const auto & roadmapOMPL = p.getRoadmap();
    int n_vertices = boost::num_vertices(roadmapOMPL);
    std::cout << "n_vertices " << n_vertices << std::endl;
    int n_edges = boost::num_edges(roadmapOMPL);
    std::cout << "n_edges " << n_edges << std::endl;
    // // const auto& roadmapOMPL = p.getDenseGraph();
    auto stateProperty = boost::get(
      geometric::PRMstar::vertex_state_t(), roadmapOMPL);

    // output
    boost::python::list edges;

    // read the edges
    BOOST_FOREACH(
      const geometric::PRMstar::Edge e, boost::edges(
        roadmapOMPL))
    {
      int i = e.m_source;
      int j = e.m_target;

      if(i >= n_vertices || j >= n_vertices) {
        continue;
      }
      if(boost::python::len(edges) >= 100) {
        break;
      }

      std::cout << "i: " << i << " j: " << j << std::endl;

      base::State * state_i = stateProperty[i];
      base::State * state_j = stateProperty[j];
      if (state_i && state_j) {
        const auto typedState_i =
          state_i->as<base::RealVectorStateSpace::StateType>();
        float x_i = (*typedState_i)[0];
        float y_i = (*typedState_i)[1];

        const auto typedState_j =
          state_j->as<base::RealVectorStateSpace::StateType>();
        float x_j = (*typedState_j)[0];
        float y_j = (*typedState_j)[1];

        std::cout << "x_i: " << x_i << " y_i: " << y_i << " x_j: " << x_j << " y_j: " << y_j << std::endl;

        edges.append(boost::python::make_tuple(i, x_i, y_i, j, x_j, y_j));
      } else {
        std::cout << state_i << " " << state_j << std::endl;
      }
    }


    return boost::python::make_tuple(edges, duration_ms);

  }
};

BOOST_PYTHON_MODULE(libomplpy)
{
  using namespace boost::python;
  namespace bp = boost::python;
  class_<Ompl>("Ompl", init<>())
  .def(
    "runSparsTwo", &Ompl::runSparsTwo,
    (
      bp::arg("mapFile"),
      bp::arg("seed"),
      bp::arg("denseDelta"),
      bp::arg("sparseDelta"),
      bp::arg("stretchFactor"),
      bp::arg("maxFailures"),
      bp::arg("maxTime"),
      bp::arg("maxIter"))
  ).def(
    "runPRMStar", &Ompl::runPRMStar,
    (
      bp::arg("mapFile"),
      bp::arg("seed"),
      bp::arg("denseDelta"),
      bp::arg("sparseDelta"),
      bp::arg("stretchFactor"),
      bp::arg("maxFailures"),
      bp::arg("maxTime"),
      bp::arg("maxIter"))
  );
}
