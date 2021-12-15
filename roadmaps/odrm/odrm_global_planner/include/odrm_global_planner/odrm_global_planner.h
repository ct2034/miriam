#ifndef ADAMSMAP_GLOBAL_PLANNER_CPP
#define ADAMSMAP_GLOBAL_PLANNER_CPP

#include <mutex>
#include <cmath>

#include <ros/ros.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/costmap_2d.h>
#include <nav_core/base_global_planner.h>

#include <graph_msgs/Edges.h>
#include <graph_msgs/GeometryGraph.h>
#include <geometry_msgs/PoseStamped.h>
#include <angles/angles.h>

#include <base_local_planner/world_model.h>
#include <base_local_planner/costmap_model.h>

#include <flann/flann.hpp>
#include <boost/graph/astar_search.hpp>
#include <boost/graph/adjacency_list.hpp>

using namespace boost;

namespace odrm_global_planner
{
static const int MAX_N{ 1024 };
static const int DIMENSIONS{ 2 };
static const float EPSILON{ 1E-3 };
static const uint DEFAULT_POSES_PER_METER{ 50 };

inline bool operator==(const geometry_msgs::PoseStamped& a, const geometry_msgs::Point& b)
{
  return (std::fabs(a.pose.position.x - b.x) < EPSILON) && (std::fabs(a.pose.position.y - b.y) < EPSILON);
}

inline bool operator==(const geometry_msgs::PoseStamped& a, const geometry_msgs::PoseStamped& b)
{
  return (std::fabs(a.pose.position.x - b.pose.position.x) < EPSILON) &&
         (std::fabs(a.pose.position.y - b.pose.position.y) < EPSILON);
}

inline bool operator!=(const geometry_msgs::PoseStamped& a, const geometry_msgs::PoseStamped& b)
{
  return !(a == b);
}

inline float operator-(const geometry_msgs::PoseStamped& a, const geometry_msgs::PoseStamped& b)
{
  return sqrt(pow(a.pose.position.x - b.pose.position.x, 2) + pow(a.pose.position.y - b.pose.position.y, 2));
}

inline double toYaw(const geometry_msgs::Quaternion& q)
{
  tf2::Quaternion tq;
  tf2::convert(q, tq);
  return (std::fmod(tq.getAngle() + M_PI, 2 * M_PI)) - M_PI;
}

inline std::ostream& operator<<(std::ostream& output, const geometry_msgs::PoseStamped& x)
{
  output << "(x: " << x.pose.position.x << ", y: " << x.pose.position.y << ", yaw: " << toYaw(x.pose.orientation)
         << ")";
  return output;
}

inline double toYaw(const geometry_msgs::PoseStamped& a)
{
  return toYaw(a.pose.orientation);
}

inline geometry_msgs::Quaternion yawToQuaternion(double yaw)
{
  tf2::Quaternion tq;
  tq.setRPY(0, 0, yaw);
  geometry_msgs::Quaternion q;
  tf2::convert(tq, q);
  return q;
}

/**
 * @class OdrmGlobalPlanner
 * @brief Provides a simple global planner that will compute a valid goal point for the local planner by walking back
 * along the vector between the robot and the user-specified goal point until a valid cost is found.
 */
class OdrmGlobalPlanner : public nav_core::BaseGlobalPlanner
{
public:
  /**
   * @brief  Constructor for the OdrmGlobalPlanner
   */
  OdrmGlobalPlanner();
  /**
   * @brief  Constructor for the OdrmGlobalPlanner
   * @param  name The name of this planner
   * @param  costmap_ros A pointer to the ROS wrapper of the costmap to use for planning
   */
  OdrmGlobalPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros);

  /**
   * @brief  Initialization function for the OdrmGlobalPlanner
   * @param  name The name of this planner
   * @param  costmap_ros A pointer to the ROS wrapper of the costmap to use for planning
   */
  void initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros);

  /**
   * @brief Given a goal pose in the world, compute a plan
   * @param start The start pose
   * @param goal The goal pose
   * @param plan The plan... filled by the planner
   * @return True if a valid plan was found, false otherwise
   */
  bool makePlan(const geometry_msgs::PoseStamped& start, const geometry_msgs::PoseStamped& goal,
                std::vector<geometry_msgs::PoseStamped>& plan);

  void roadmapCb(const graph_msgs::GeometryGraph& gg);

  geometry_msgs::PoseStamped poseStampedFromPoint(geometry_msgs::Point& p);
  geometry_msgs::PoseStamped poseStampedFromXY(float x, float y);

  int findIndexOnPrevTrack(flann::Matrix<int>& indices);

  float plan_boost(std::vector<geometry_msgs::PoseStamped>& plan, int start_idx, int goal_idx,
                   const geometry_msgs::PoseStamped& start, const geometry_msgs::PoseStamped& goal);
  double get_angle(const geometry_msgs::PoseStamped& a, geometry_msgs::PoseStamped& b, geometry_msgs::PoseStamped& o);
  void make_poses_along_plan(std::vector<geometry_msgs::PoseStamped>& in, std::vector<geometry_msgs::PoseStamped>& out);

private:
  costmap_2d::Costmap2DROS* costmap_ros_;
  costmap_2d::Costmap2D* costmap_;
  base_local_planner::WorldModel* world_model_;  ///< @brief The world model that the controller will use
  ros::Subscriber roadmap_sub_;
  ros::Publisher path_pub_;
  bool graph_received_{ false };
  graph_msgs::GeometryGraph graph_;
  std::mutex graph_guard_;
  int poses_per_meter_;

  // flann
  flann::Matrix<float> dataset = flann::Matrix<float>(new float[MAX_N * DIMENSIONS], MAX_N, DIMENSIONS);
  flann::Index<flann::L2<float>> flann_index;

  // boost graph
  typedef adjacency_list<listS, vecS, directedS, no_property, property<edge_weight_t, float>> mygraph_t;
  typedef property_map<mygraph_t, edge_weight_t>::type WeightMap;
  typedef std::vector<geometry_msgs::Point> LocMap;
  typedef mygraph_t::vertex_descriptor vertex;
  typedef mygraph_t::edge_descriptor edge_descriptor;
  typedef mygraph_t::vertex_iterator vertex_iterator;

  mygraph_t g;
  WeightMap weightmap;
  LocMap locations;

  /**
   * @brief  Checks the legality of the robot footprint at a position and orientation using the world model
   * @param x_i The x position of the robot
   * @param y_i The y position of the robot
   * @param theta_i The orientation of the robot
   * @return
   */
  double footprintCost(double x_i, double y_i, double theta_i);

  bool initialized_;
};
};      // namespace odrm_global_planner
#endif  // ADAMSMAP_GLOBAL_PLANNER_CPP
