# Gray-Scott optimized roadmap (GSORM)

## Requirements

- Boost 1.71
- lodepng (in folder)
<!-- - pytorch for cpp as in https://pytorch.org/cppdocs/installing.html 
    - memorize the folder
    - in CMakelists.txt, change the path in "set(CMAKE_PREFIX_PATH ..." to the folder you memorized -->
- opencv 4.2.0

## build

```
mkdir build
cd build
cmake ..
make
```