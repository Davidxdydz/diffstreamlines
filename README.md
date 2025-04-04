# Installation
1. clone the repo
2. have the basics installed (pybind11, torch, wheel)
3. in this dir:
```bash
pip install .
```
4.
```python
import diffstreamlines
paths, pathlengths = diffstreamlines(velocities,start_positions,dt,steps)
```
5. see [test.ipynb](test.ipynb) for example usage