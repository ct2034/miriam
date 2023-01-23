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

struct GenerationType {
  enum Type
  {
    SPARS,
    SPARS2,
  };
};

std::istream& operator>>(std::istream& in, GenerationType::Type& type)
{
  std::string token;
  in >> token;
  if (token == "SPARS")
    type = GenerationType::SPARS;
  else if (token == "SPARS2")
    type = GenerationType::SPARS2;
  else
    in.setstate(std::ios_base::failbit);
  return in;
}

class ImageStateValidityChecker : public ompl::base::StateValidityChecker
{
public:
  ImageStateValidityChecker(
    ompl::base::SpaceInformationPtr si,
    const std::vector<unsigned char>& image,
    unsigned width,
    unsigned height)
    : StateValidityChecker(si)
    , image_(image)
    , width_(width)
    , height_(height)
  {
  }

  bool isValid(const ompl::base::State* state) const
  {
    const ompl::base::RealVectorStateSpace::StateType* typedState =
        state->as<ompl::base::RealVectorStateSpace::StateType>();

    int x = (int)(*typedState)[0];
    int y = (int)(*typedState)[1];

    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
      return false;
    }

    return image_[y*width_ + x] > 200;
  }

private:
  const std::vector<unsigned char>& image_;
  unsigned width_;
  unsigned height_;
};

int main(int argc, char** argv) {


  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string environmentFile;
  std::string outputFile;
  std::string configFile;
  GenerationType::Type genType = GenerationType::SPARS;
  const size_t dimension = 2;
  desc.add_options()
      ("help", "produce help message")
      ("environment,e", po::value<std::string>(&environmentFile)->required(), "input file for environment (PNG)")
      ("output,o", po::value<std::string>(&outputFile)->required(), "output file for graph (YAML)")
      ("config,c", po::value<std::string>(&configFile)->required(), "config file for advanced parameters (YAML)")
      ("type", po::value<GenerationType::Type>(&genType)->default_value(GenerationType::SPARS)->multitoken(), "Method, one of [SPARS,SPARS2]. Default: SPARS")
  ;

  po::variables_map vm;
  try
  {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }
  }
  catch(po::error& e)
  {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }

  // Load PNG image

  std::vector<unsigned char> image;  // the raw pixels
  unsigned width, height;

  // decode
  unsigned error = lodepng::decode(image, width, height, environmentFile, LCT_GREY, 8);

  // if there's an error, display it
  if (error) {
    std::cerr << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    return 1;
  }

  std::cout << "Loaded environment: " << width << "x" << height << std::endl;

  YAML::Node cfg = YAML::LoadFile(configFile);

  if (genType == GenerationType::SPARS
        || genType == GenerationType::SPARS2) {
    base::StateSpacePtr space(new base::RealVectorStateSpace(dimension));
    base::RealVectorBounds bounds(dimension);
    bounds.setLow(0, 0);
    bounds.setHigh(0, width);
    bounds.setLow(1, 0);
    bounds.setHigh(1, height);
    space->as<base::RealVectorStateSpace>()->setBounds(bounds);

    base::SpaceInformationPtr si(new base::SpaceInformation(space));
    base::StateValidityCheckerPtr stateValidityChecker(new ImageStateValidityChecker(si, image, width, height));
    si->setStateValidityChecker(stateValidityChecker);
    si->setStateValidityCheckingResolution(0.01); // 1%
    si->setup();

    base::ProblemDefinitionPtr pdef(new base::ProblemDefinition(si));
    base::ScopedState<base::RealVectorStateSpace> start(space);
    // fill start state
    base::ScopedState<base::RealVectorStateSpace> goal(space);
    // fill goal state
    pdef->setStartAndGoalStates(start, goal);

    if (genType == GenerationType::SPARS) {
      auto node = cfg["spars"];

      geometric::SPARS p(si);
      p.setProblemDefinition(pdef);
      // std::cout << p.getDenseDeltaFraction()
      // << "Delta: " << p.getSparseDeltaFraction() * si->getMaximumExtent()
      // << " " << p.getStretchFactor()
      // << " " << p.getMaxFailures() << std::endl;
      // p.setSparseDeltaFraction(1.0 / si->getMaximumExtent() );

      p.setDenseDeltaFraction(node["denseDelta"].as<float>() / si->getMaximumExtent() );
      p.setSparseDeltaFraction(node["sparseDelta"].as<float>() / si->getMaximumExtent() );
      p.setStretchFactor(node["stretchFactor"].as<float>());
      p.setMaxFailures(node["maxFailures"].as<int>());

      p.constructRoadmap(
        base::timedPlannerTerminationCondition(node["maxTime"].as<double>()), true);
        // /*base::timedPlannerTerminationCondition(30)*/base::IterationTerminationCondition(node["maxIter"].as<int>()), true);

      p.printDebug();
     
      const auto& roadmapOMPL = p.getRoadmap();
      std::cout << "#vertices " << boost::num_vertices(roadmapOMPL) << std::endl;
      std::cout << "#edges " << boost::num_edges(roadmapOMPL) << std::endl;
      // // const auto& roadmapOMPL = p.getDenseGraph();
      auto stateProperty = boost::get(geometric::SPARS::vertex_state_t(), roadmapOMPL);

      // for (size_t i = 0; i < boost::num_vertices(roadmapOMPL); ++i) {
      //   base::State* state = stateProperty[i];
      //   if (state) {
      //     const base::RealVectorStateSpace::StateType* typedState = state->as<base::RealVectorStateSpace::StateType>();
      //     float x = (*typedState)[0];
      //     float y = (*typedState)[1];
      //     std::cout << "v" << i << "," << x << "," << y << std::endl;
      //   }
      // }

      // output
      std::ofstream stream(outputFile.c_str());

      BOOST_FOREACH (const geometric::SPARS::SparseEdge e, boost::edges(roadmapOMPL)) {
        size_t i = boost::source(e, roadmapOMPL);
        size_t j = boost::target(e, roadmapOMPL);

        base::State* state_i = stateProperty[i];
        base::State* state_j = stateProperty[j];
        if (state_i && state_j) {
          const auto typedState_i = state_i->as<base::RealVectorStateSpace::StateType>();
          float x_i = (*typedState_i)[0];
          float y_i =(*typedState_i)[1];

          const auto typedState_j = state_j->as<base::RealVectorStateSpace::StateType>();
          float x_j = (*typedState_j)[0];
          float y_j = (*typedState_j)[1];

          stream << x_i << "," << y_i << "," <<  x_j << "," << y_j << std::endl;
        }
      }
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

      p.constructRoadmap(/*base::timedPlannerTerminationCondition(30)*/base::IterationTerminationCondition(node["maxIter"].as<int>()), true);

      const auto& roadmapOMPL = p.getRoadmap();
      auto stateProperty = boost::get(geometric::SPARStwo::vertex_state_t(), roadmapOMPL);

      std::unordered_map<geometric::SPARS::SparseVertex, vertex_t> vertexMap;
      for (size_t i = 0; i < boost::num_vertices(roadmapOMPL); ++i) {
        base::State* state = stateProperty[i];
        if (state) {
          position_t p (-1, -1, -1);
          const base::RealVectorStateSpace::StateType* typedState = state->as<base::RealVectorStateSpace::StateType>();
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
      BOOST_FOREACH (const geometric::SPARStwo::Edge e, boost::edges(roadmapOMPL)) {
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
  BOOST_FOREACH (const edge_t e, boost::edges(roadmap)) {
    roadmap[e].name = "e" + std::to_string(i);
    ++i;
  }

  saveSearchGraph(roadmap, outputFile);
#endif
  return 0;
}

