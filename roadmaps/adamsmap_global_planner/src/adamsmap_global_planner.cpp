#include <adamsmap_global_planner/adamsmap_global_planner.h>
#include <pluginlib/class_list_macros.h>
#include <tf2/convert.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// register this planner as a BaseGlobalPlanner plugin
PLUGINLIB_EXPORT_CLASS(adamsmap_global_planner::AdamsmapGlobalPlanner, nav_core::BaseGlobalPlanner)

namespace adamsmap_global_planner
{
AdamsmapGlobalPlanner::AdamsmapGlobalPlanner()
  : costmap_ros_(NULL), initialized_(false), flann_index(flann::LinearIndexParams())
{
}

AdamsmapGlobalPlanner::AdamsmapGlobalPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros)
  : costmap_ros_(NULL), initialized_(false), flann_index(flann::LinearIndexParams())
{
  initialize(name, costmap_ros);
}

void AdamsmapGlobalPlanner::initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros)
{
  ROS_INFO("intialize");
  if (!initialized_)
  {
    costmap_ros_ = costmap_ros;
    costmap_ = costmap_ros_->getCostmap();

    ros::NodeHandle private_nh("~/" + name);
    private_nh.param("step_size", step_size_, costmap_->getResolution());
    private_nh.param("min_dist_from_robot", min_dist_from_robot_, 0.10);
    world_model_ = new base_local_planner::CostmapModel(*costmap_);

    ros::NodeHandle nh;
    roadmap_sub_ = nh.subscribe("roadmap", 1, &AdamsmapGlobalPlanner::roadmapCb, this);

    initialized_ = true;
  }
  else
    ROS_WARN("This planner has already been initialized... doing nothing");
  //    cv::vector<cv::Point2f> scheme_pts;
  //    cv::flann::Index tree= cv::flann::Index(cv::Mat(scheme_pts).reshape(1),cv::flann::LinearIndexParams());
}

// we need to take the footprint of the robot into account when we calculate cost to obstacles
double AdamsmapGlobalPlanner::footprintCost(double x_i, double y_i, double theta_i)
{
  if (!initialized_)
  {
    ROS_ERROR("The planner has not been initialized, please call initialize() to use the planner");
    return -1.0;
  }

  std::vector<geometry_msgs::Point> footprint = costmap_ros_->getRobotFootprint();
  // if we have no footprint... do nothing
  if (footprint.size() < 3)
    return -1.0;

  // check if the footprint is legal
  double footprint_cost = world_model_->footprintCost(x_i, y_i, theta_i, footprint);
  return footprint_cost;
}

bool AdamsmapGlobalPlanner::makePlan(const geometry_msgs::PoseStamped& start, const geometry_msgs::PoseStamped& goal,
                                     std::vector<geometry_msgs::PoseStamped>& plan)
{
  ROS_INFO("makePlan");

  if (!initialized_)
  {
    ROS_ERROR("The planner has not been initialized, please call initialize() to use the planner");
    return false;
  }

  if (!graph_received_)
  {
    ROS_ERROR("We have not yet received a roadmap graph!");
    return false;
  }
  std::lock_guard<std::mutex> guard(graph_guard_);

  ROS_DEBUG("Got a start: %.2f, %.2f, and a goal: %.2f, %.2f", start.pose.position.x, start.pose.position.y,
            goal.pose.position.x, goal.pose.position.y);

  plan.clear();
  costmap_ = costmap_ros_->getCostmap();

  if (goal.header.frame_id != costmap_ros_->getGlobalFrameID())
  {
    ROS_ERROR("This planner as configured will only accept goals in the %s frame, but a goal was sent in the %s frame.",
              costmap_ros_->getGlobalFrameID().c_str(), goal.header.frame_id.c_str());
    return false;
  }

  size_t nn = 1;
  size_t n_query = 2;
  flann::Matrix<float> query(new float[n_query * dimensions], n_query, dimensions);
  query[0][0] = start.pose.position.x;
  query[0][1] = start.pose.position.y;
  query[1][0] = goal.pose.position.x;
  query[1][1] = goal.pose.position.y;
  flann::Matrix<int> indices(new int[n_query * nn], n_query, nn);
  flann::Matrix<float> dists(new float[n_query * nn], n_query, nn);

  // Search
  flann::Logger::setLevel(flann::FLANN_LOG_INFO);  // fix  https://github.com/mariusmuja/flann/issues/198
  flann_index.knnSearch(query, indices, dists, nn, flann::SearchParams(64));

  int start_idx = indices[0][0];
  int goal_idx = indices[0][1];

  ROS_DEBUG_STREAM("start_idx " << start_idx);
  ROS_DEBUG_STREAM("goal_idx  " << goal_idx);

  return true;
}

void AdamsmapGlobalPlanner::roadmapCb(const graph_msgs::GeometryGraph& gg)
{
  ROS_DEBUG("received a roadmap");
  std::lock_guard<std::mutex> guard(graph_guard_);
  graph_ = gg;
  graph_received_ = true;

  int n = gg.nodes.size();
  dataset = flann::Matrix<float>(new float[n * dimensions], n, dimensions);
  for (int row = 0; row < n; row++)
  {
    dataset[row][0] = gg.nodes[row].x;
    dataset[row][1] = gg.nodes[row].y;
  }
  flann_index.buildIndex(dataset);
  ROS_DEBUG("flann index built");
}
};  // namespace adamsmap_global_planner
