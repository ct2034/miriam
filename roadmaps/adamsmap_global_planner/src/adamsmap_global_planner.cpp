#include <adamsmap_global_planner/adamsmap_global_planner.h>
#include <pluginlib/class_list_macros.h>
#include <tf2/convert.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nav_msgs/Path.h>

// register this planner as a BaseGlobalPlanner plugin
PLUGINLIB_EXPORT_CLASS(adamsmap_global_planner::AdamsmapGlobalPlanner, nav_core::BaseGlobalPlanner)

using namespace boost;

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
  ROS_DEBUG_STREAM("name: " << name);
  if (!initialized_)
  {
    costmap_ros_ = costmap_ros;
    costmap_ = costmap_ros_->getCostmap();

    ros::NodeHandle private_nh("~/" + name);
    if (!ros::param::get("poses_per_meter", poses_per_meter_))
    {
      poses_per_meter_ = DEFAULT_POSES_PER_METER;
      ROS_DEBUG("Param ~poses_per_meter not found using default value %d", poses_per_meter_);
    }
    world_model_ = new base_local_planner::CostmapModel(*costmap_);
    path_pub_ = private_nh.advertise<nav_msgs::Path>("path", 1);

    ros::NodeHandle nh;
    roadmap_sub_ = nh.subscribe("roadmap", 1, &AdamsmapGlobalPlanner::roadmapCb, this);

    initialized_ = true;
  }
  else
    ROS_WARN("This planner has already been initialized... doing nothing");
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

// distance heuristic
template <class Graph, class CostType, class LocMap>
class distance_heuristic : public astar_heuristic<Graph, CostType>
{
public:
  typedef typename graph_traits<Graph>::vertex_descriptor Vertex;

  distance_heuristic(LocMap l, Vertex goal) : m_location(l), m_goal(goal)
  {
  }

  CostType operator()(Vertex u)
  {
    CostType dx = m_location[m_goal].x - m_location[u].x;
    CostType dy = m_location[m_goal].y - m_location[u].y;
    return ::sqrt(dx * dx + dy * dy);
  }

private:
  LocMap m_location;
  Vertex m_goal;
};

class found_goal
{
public:
  explicit found_goal(int goal)
  {
    actual_goal = goal;
  }
  int actual_goal;
};  // exception for termination

// visitor that terminates when we find the goal
template <class Vertex>
class astar_goal_visitor : public boost::default_astar_visitor
{
public:
  astar_goal_visitor(int goal)
  {
    _goal = goal;
  }

  template <class Graph>
  void examine_vertex(Vertex u, Graph& g)
  {
    if (u == _goal)
      throw found_goal((int)u);
  }

private:
  int _goal;
};

float AdamsmapGlobalPlanner::plan_boost(std::vector<geometry_msgs::PoseStamped>& plan, int start_idx, int goal_idx,
                                        const geometry_msgs::PoseStamped& start, const geometry_msgs::PoseStamped& goal)
{
  std::vector<mygraph_t::vertex_descriptor> p(num_vertices(g));
  std::vector<float> d(num_vertices(g));

  try
  {
    ROS_DEBUG("astar_search ...");
    // call astar named parameter interface
    astar_search(g, start_idx, distance_heuristic<mygraph_t, float, LocMap>(locations, goal_idx),
                 predecessor_map(&p[0]).distance_map(&d[0]).visitor(astar_goal_visitor<vertex>((int)goal_idx)));
  }
  catch (found_goal& fg)
  {  // found a path to the goal
    ROS_DEBUG_STREAM("Path found");
    plan.clear();
    plan.push_back(goal);
    vertex v;
    for (v = (vertex)fg.actual_goal;; v = p[v])
    {
      plan.push_back(poseStampedFromPoint(locations[v]));
      if (p[v] == v)
        break;
    }

    while (plan.size() >= 2 ? M_PI_2 / 2 > get_angle(start, *(plan.end() - 1), *(plan.end())) : false)
      plan.pop_back();
    if (plan.size() == 2)  // if only goal and one point is in there, go directly
      plan.pop_back();
    plan.push_back(start);
  }
  catch (exception& e)
  {
    ROS_ERROR("Exception");
  }
  bool first = true;
  double len = 0;
  geometry_msgs::PoseStamped previous;
  for (auto& p : plan)
  {
    if (!first)
    {
      len += p - previous;
    }
    previous = geometry_msgs::PoseStamped(p);
    first = false;
  }
  return len;
}

double AdamsmapGlobalPlanner::get_angle(const geometry_msgs::PoseStamped& a, geometry_msgs::PoseStamped& b,
                                        geometry_msgs::PoseStamped& o)
{
  double la = a - o;
  double lb = b - o;
  double dot = ((a.pose.position.x - o.pose.position.x) / la * (b.pose.position.x - o.pose.position.x) / lb +
                (a.pose.position.y - o.pose.position.y) / la * (b.pose.position.y - o.pose.position.y) / lb);

  dot = (dot < -1.0 ? -1.0 : (dot > 1.0 ? 1.0 : dot));
  return acos(dot);
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

  ROS_DEBUG("Got a start: %.2f, %.2f, %.2f, and a goal: %.2f, %.2f, %.2f", start.pose.position.x, start.pose.position.y,
            toYaw(start.pose.orientation), goal.pose.position.x, goal.pose.position.y, toYaw(goal.pose.orientation));

  plan.clear();
  costmap_ = costmap_ros_->getCostmap();

  if (goal.header.frame_id != costmap_ros_->getGlobalFrameID())
  {
    ROS_ERROR("This planner as configured will only accept goals in the %s frame, but a goal was sent in the %s frame.",
              costmap_ros_->getGlobalFrameID().c_str(), goal.header.frame_id.c_str());
    return false;
  }

  size_t nn = 2;
  size_t n_query = 2;
  flann::Matrix<float> query(new float[n_query * DIMENSIONS], n_query, DIMENSIONS);
  query[0][0] = start.pose.position.x;
  query[0][1] = start.pose.position.y;
  query[1][0] = goal.pose.position.x;
  query[1][1] = goal.pose.position.y;
  flann::Matrix<int> indices(new int[n_query * nn], n_query, nn);
  flann::Matrix<float> dists(new float[n_query * nn], n_query, nn);

  // Search
  flann::Logger::setLevel(flann::FLANN_LOG_INFO);  // fix  https://github.com/mariusmuja/flann/issues/198
  flann_index.knnSearch(query, indices, dists, nn, flann::SearchParams(64));

  ROS_DEBUG_STREAM("locations.size()    " << locations.size());
  ROS_DEBUG_STREAM("g.m_edges.size()    " << g.m_edges.size());
  ROS_DEBUG_STREAM("g.m_vertices.size() " << g.m_vertices.size());

  // Boost
  std::vector<geometry_msgs::PoseStamped> vertex_plan = std::vector<geometry_msgs::PoseStamped>(0);
  float min_cost = std::numeric_limits<float>::max();
  for (int i_c = 0; i_c < pow(nn, 2); i_c++)
  {
    std::vector<geometry_msgs::PoseStamped> tmp_plan = std::vector<geometry_msgs::PoseStamped>(0);
    size_t s = i_c % nn;
    size_t g = (i_c - s) / nn;
    float cost = plan_boost(tmp_plan, indices[0][s], indices[1][g], start, goal);
    if (tmp_plan.size() == 0)
      cost = std::numeric_limits<float>::max();
    ROS_DEBUG_STREAM("cost " << cost);
    if (cost < min_cost)
    {
      min_cost = cost;
      vertex_plan.clear();
      std::copy(std::begin(tmp_plan), std::end(tmp_plan), std::back_inserter(vertex_plan));
    }
  }
  ROS_DEBUG_STREAM("min_cost " << min_cost);

  std::reverse(std::begin(vertex_plan), std::end(vertex_plan));
  ROS_DEBUG_STREAM("vertex_plan.size() " << vertex_plan.size());

  plan.clear();
  make_poses_along_plan(vertex_plan, plan);
  plan.push_back(goal);
  ROS_DEBUG_STREAM("plan.size() " << plan.size());

  // viz
  nav_msgs::Path path_viz;
  path_viz.header.frame_id = "map";
  path_viz.header.stamp = ros::Time::now();
  std::copy(std::begin(plan), std::end(plan), std::back_inserter(path_viz.poses));
  path_pub_.publish(path_viz);

  for (auto it = plan.begin(); it != plan.end(); ++it)
  {
    if (it != plan.begin())
    {
      if (*it - *(it - 1) > 2.0 / poses_per_meter_)
      {
        ROS_DEBUG_STREAM("d: " << *it - *(it - 1) << " between " << *it << " and " << *(it - 1));
      }
      //      ROS_ASSERT(*it - *(it - 1) <= 2.0 / poses_per_meter_);
    }
  }

  return true;
}

void AdamsmapGlobalPlanner::make_poses_along_plan(std::vector<geometry_msgs::PoseStamped>& in,
                                                  std::vector<geometry_msgs::PoseStamped>& out)
{
  //  ROS_ASSERT(in.size() >= 2);
  for (auto planit = in.begin(); planit != in.end(); ++planit)
  {
    if (planit == in.begin())
    {
      out.push_back(*planit);
    }
    else  // everything after begin
    {
      auto prev = planit - 1;
      double dx = planit->pose.position.x - prev->pose.position.x;
      double dy = planit->pose.position.y - prev->pose.position.y;
      double d = std::sqrt(pow(dx, 2) + pow(dy, 2));
      double yaw = std::atan2(dy, dx);
      int n = std::ceil(float(poses_per_meter_) / d);
      ROS_DEBUG_STREAM("dx: " << dx << " d: " << d << " yaw: " << yaw << " n: " << n);
      if (d == 0)
      {
        out.push_back(*planit);
      }
      else
      {
        for (int i = 0; i < n; i++)
        {
          geometry_msgs::PoseStamped p;
          p.pose.position.x = prev->pose.position.x + float(i + 1) / n * dx;
          p.pose.position.y = prev->pose.position.y + float(i + 1) / n * dy;
          p.pose.orientation = yawToQuaternion(yaw);
          out.push_back(p);
        }
      }
    }
  }
  ROS_DEBUG_STREAM("out.size(): " << out.size() << " in.size(): " << in.size());
  //  ROS_ASSERT(out.size() >= in.size());
}

geometry_msgs::PoseStamped AdamsmapGlobalPlanner::poseStampedFromPoint(geometry_msgs::Point& p)
{
  geometry_msgs::PoseStamped ps;
  ps.header.frame_id = "map";
  ps.pose.position.x = p.x;
  ps.pose.position.y = p.y;
  ps.pose.orientation.w = 1;
  return ps;
}

geometry_msgs::PoseStamped AdamsmapGlobalPlanner::poseStampedFromXY(float x, float y)
{
  geometry_msgs::Point p;
  p.x = x;
  p.y = y;
  return poseStampedFromPoint(p);
}

void AdamsmapGlobalPlanner::roadmapCb(const graph_msgs::GeometryGraph& gg)
{
  ROS_DEBUG("received a roadmap");
  std::lock_guard<std::mutex> guard(graph_guard_);
  graph_ = gg;
  graph_received_ = true;

  int n = gg.nodes.size();
  dataset = flann::Matrix<float>(new float[n * DIMENSIONS], n, DIMENSIONS);
  for (int row = 0; row < n; row++)
  {
    dataset[row][0] = gg.nodes[row].x;
    dataset[row][1] = gg.nodes[row].y;
  }
  flann_index.buildIndex(dataset);
  ROS_DEBUG("flann index built");

  g = mygraph_t(n);
  weightmap = get(edge_weight, g);
  locations = gg.nodes;

  for (std::size_t i = 0; i < n; i++)
  {
    graph_msgs::Edges es = gg.edges[i];
    for (std::size_t j = 0; j < es.weights.size(); j++)
    {
      edge_descriptor e;
      bool inserted;
      tie(e, inserted) = add_edge(i, es.node_ids[j], g);
      weightmap[e] = es.weights[j];
    }
  }
  ROS_DEBUG("planning graph saved ...");

  ROS_DEBUG_STREAM("g.m_edges.size()    " << g.m_edges.size());
  ROS_DEBUG_STREAM("g.m_vertices.size() " << g.m_vertices.size());
}
};  // namespace adamsmap_global_planner
