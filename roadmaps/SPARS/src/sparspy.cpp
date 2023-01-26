#include <boost/python.hpp>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

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

// LodePNG
#include "lodepng.h"

using namespace std;
using namespace ompl;

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
    ompl::base::SpaceInformationPtr si, const std::vector<unsigned char> & image, unsigned width,
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

    return image_[y * width_ + x] > 200;
  }

private:
  const std::vector<unsigned char> & image_;
  unsigned width_;
  unsigned height_;
};

class Spars
{
public:
  Spars()
  {
  }

  boost::python::list run(
    std::string mapFile, int seed,
    float denseDelta, float sparseDelta, float stretchFactor,
    int maxFailures, double maxTime)
  {
    ompl::RNG::setSeed(seed);

    GenerationType::Type genType = GenerationType::SPARS;
    const size_t dimension = 2;

    std::vector<unsigned char> image;  // the raw pixels
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

    if (genType == GenerationType::SPARS || genType == GenerationType::SPARS2) {
      base::StateSpacePtr space(new base::RealVectorStateSpace(dimension));
      base::RealVectorBounds bounds(dimension);
      bounds.setLow(0, 0);
      bounds.setHigh(0, width);
      bounds.setLow(1, 0);
      bounds.setHigh(1, height);
      space->as<base::RealVectorStateSpace>()->setBounds(bounds);

      base::SpaceInformationPtr si(new base::SpaceInformation(space));
      base::StateValidityCheckerPtr stateValidityChecker(new ImageStateValidityChecker(
          si, image,
          width, height));
      si->setStateValidityChecker(stateValidityChecker);
      si->setStateValidityCheckingResolution(0.01);  // 1%
      si->setup();

      base::ProblemDefinitionPtr pdef(new base::ProblemDefinition(si));
      base::ScopedState<base::RealVectorStateSpace> start(space);
      // fill start state
      base::ScopedState<base::RealVectorStateSpace> goal(space);
      // fill goal state
      // pdef->setStartAndGoalStates(start, goal);  // TODO: why this?

      if (genType == GenerationType::SPARS) {
        geometric::SPARS p(si);
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

        p.constructRoadmap(
          base::timedPlannerTerminationCondition(maxTime), true);
        // /*base::timedPlannerTerminationCondition(30)*/base::IterationTerminationCondition(node["maxIter"].as<int>()),
        // true);

        p.printDebug();

        const auto & roadmapOMPL = p.getRoadmap();
        std::cout << "#vertices " << boost::num_vertices(roadmapOMPL) << std::endl;
        std::cout << "#edges " << boost::num_edges(roadmapOMPL) << std::endl;
        // // const auto& roadmapOMPL = p.getDenseGraph();
        auto stateProperty = boost::get(geometric::SPARS::vertex_state_t(), roadmapOMPL);

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

        return edges;
      }
#if 0
      if (genType == GenerationType::SPARS2) {
        auto node = cfg["spars2"];

        geometric::SPARStwo p(si);
        p.setProblemDefinition(pdef);

        p.setDenseDeltaFraction(node["denseDelta"].as<float>() / si->getMaximumExtent() );
        p.setSparseDeltaFraction(node["sparseDelta"].as<float>() / si->getMaximumExtent() );
        p.setStretchFactor(node["stretchFactor"].as<float>());
        p.setMaxFailures(node["maxFailures"].as<int>());

        p.constructRoadmap(
          /*base::timedPlannerTerminationCondition(30)*/ base::IterationTerminationCondition(
            node["maxIter"].as<int>()), true);

        const auto & roadmapOMPL = p.getRoadmap();
        auto stateProperty = boost::get(geometric::SPARStwo::vertex_state_t(), roadmapOMPL);

        std::unordered_map<geometric::SPARS::SparseVertex, vertex_t> vertexMap;
        for (size_t i = 0; i < boost::num_vertices(roadmapOMPL); ++i) {
          base::State * state = stateProperty[i];
          if (state) {
            position_t p(-1, -1, -1);
            const base::RealVectorStateSpace::StateType * typedState =
              state->as<base::RealVectorStateSpace::StateType>();
            if (dimension == 3) {
              p = position_t((*typedState)[0], (*typedState)[1], (*typedState)[2]);
            } else if (dimension == 2) {
              p = position_t((*typedState)[0], (*typedState)[1], fixedZ);
            }
            auto v = boost::add_vertex(roadmap);
            roadmap[v].pos = p;
            roadmap[v].name = "v" + std::to_string(v);
            vertexMap[i] = v;
          }
        }

        // add OMPL edges
        BOOST_FOREACH(const geometric::SPARStwo::Edge e, boost::edges(roadmapOMPL)) {
          size_t i = boost::source(e, roadmapOMPL);
          size_t j = boost::target(e, roadmapOMPL);

          add_edge(vertexMap.at(i), vertexMap.at(j), roadmap);
        }
      }
#endif
    }
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
    bp::arg("stretchFactor"), bp::arg("maxFailures"), bp::arg("maxTime")))
  ;
}
