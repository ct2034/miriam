# Generate Roadmaps using SPARS/SPARS2

This tool uses the official SPARS(2) implementation to generate a simple roadmap on 2D PNG images

## Building

```
mkdir build
cd build
cmake ..
make
```

## Execution

```
cd build
./generateRoadmap -c ../examples/config.yaml -e ../../odrm_eval/maps/o.png -o test.csv
python3 ../examples/visualizeRoadmap.py test.csv ../../odrm_eval/maps/o.png
```

