#include <boost/python.hpp>
#include <iostream>
extern "C" char const* greet()
{
  return "hello, world";
}

BOOST_PYTHON_MODULE(libhello_ext)
{
  using namespace boost::python;
  def("greet", greet);
}