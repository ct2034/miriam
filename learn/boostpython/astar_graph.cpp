//
//=======================================================================
// Copyright (c) 2004 Kristopher Beevers
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================
//

#include <boost/graph/astar_search.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/random.hpp>
#include <boost/random.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/python.hpp>
#include <ctime>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <math.h>  // for sqrt

using namespace boost;
using namespace std;

// auxiliary types
struct location
{
  float y, x;  // lat, long
};
typedef float cost;

// euclidean distance heuristic
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

struct found_goal
{
};  // exception for termination

// visitor that terminates when we find the goal
template <class Vertex>
class astar_goal_visitor : public boost::default_astar_visitor
{
public:
  astar_goal_visitor(Vertex goal) : m_goal(goal)
  {
  }
  template <class Graph>
  void examine_vertex(Vertex u, Graph& g)
  {
    if (u == m_goal)
      throw found_goal();
  }

private:
  Vertex m_goal;
};

// Input defs
typedef boost::python::list pylist;
typedef std::pair<int, int> intedge;
typedef float cost;
// Graph defs
typedef adjacency_list<listS, vecS, undirectedS, no_property, property<edge_weight_t, cost>> mygraph_t;
typedef property_map<mygraph_t, edge_weight_t>::type WeightMap;
typedef mygraph_t::vertex_descriptor vertex_t;
typedef mygraph_t::edge_descriptor edge_descriptor;

class AstarSolver
{
public:
  AstarSolver(pylist pos, pylist edges);
  pylist l;
  location retreive(int i);
  void append(int i);
  pylist plan(int start, int goal);

private:
  std::vector<location> location_v;
  std::vector<intedge> edge_v;
  std::size_t N;          // number of nodes
  std::size_t num_edges;  // number of edges
  mygraph_t g;
  WeightMap weightmap;
};

AstarSolver::AstarSolver(pylist pos, pylist edges)
{
  N = boost::python::len(pos);
  num_edges = boost::python::len(edges);
  location_v = std::vector<location>(N);
  for (std::size_t i = 0; i < N; i++)
  {
    pylist l = boost::python::extract<pylist>(pos[i]);
    location_v[i].x = boost::python::extract<double>(l[0]);
    location_v[i].y = boost::python::extract<double>(l[1]);
  }
  edge_v = std::vector<intedge>(num_edges);
  for (std::size_t i = 0; i < num_edges; i++)
  {
    pylist l = boost::python::extract<pylist>(edges[i]);
    edge_v[i].first = boost::python::extract<int>(l[0]);
    edge_v[i].second = boost::python::extract<int>(l[1]);
  }

  // create graph
  g = mygraph_t(N);
  weightmap = get(edge_weight, g);
  for (std::size_t j = 0; j < num_edges; ++j)
  {
    edge_descriptor e;
    bool inserted;
    boost::tie(e, inserted) = add_edge(edge_v[j].first, edge_v[j].second, g);
    weightmap[e] = sqrt(pow(location_v[edge_v[j].first].x - location_v[edge_v[j].second].x, 2) +
                        pow(location_v[edge_v[j].first].y - location_v[edge_v[j].second].y, 2));
  }
}

pylist AstarSolver::plan(int start_i, int goal_i)
{
  vertex_t start(start_i);
  vertex_t goal(goal_i);

  vector<mygraph_t::vertex_descriptor> p(num_vertices(g));
  vector<cost> d(num_vertices(g));
  pylist shortest_path_py;
  try
  {
    // call astar named parameter interface
    astar_search_tree(g, start, distance_heuristic<mygraph_t, cost, location*>(location_v.data(), goal),
                      predecessor_map(make_iterator_property_map(p.begin(), get(vertex_index, g)))
                          .distance_map(make_iterator_property_map(d.begin(), get(vertex_index, g)))
                          .visitor(astar_goal_visitor<vertex_t>(goal)));
  }
  catch (found_goal fg)
  {  // found a path to the goal
    list<vertex_t> shortest_path;
    for (vertex_t v = goal;; v = p[v])
    {
      shortest_path.push_front(v);
      if (p[v] == v)
        break;
    }
    list<vertex_t>::iterator spi = shortest_path.begin();
    shortest_path_py.append(start_i);
    for (++spi; spi != shortest_path.end(); ++spi)
      shortest_path_py.append(*spi);
    // shortest_path_py.append(goal_i);
  }
  return shortest_path_py;
}

location AstarSolver::retreive(int i)
{
  cout << "location of node " << i << endl;
  cout << location_v[i].x << " " << location_v[i].y << endl;
  cout << "edges" << endl;
  for (auto e : edge_v)
  {
    if (e.first == i || e.second == i)
    {
      cout << e.first << " " << e.second << endl;
    }
  }
  return location_v.data()[i];
}

void AstarSolver::append(int i)
{
  l.append(i);
}

BOOST_PYTHON_MODULE(libastar_graph)
{
  using namespace boost::python;
  class_<AstarSolver>("AstarSolver", init<pylist, pylist>())
      .def("retreive", &AstarSolver::retreive)
      .def("append", &AstarSolver::append)
      .def("plan", &AstarSolver::plan);
  class_<location>("location").def_readwrite("x", &location::x).def_readwrite("y", &location::y);
}
