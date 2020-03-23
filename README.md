# trajectory_distance
=====================

**trajectory_distance** is a Python module for computing distances between 2D-trajectory objects.
It is implemented in Cython.

## Description

9 distances between trajectories are available in the **trajectory_distance**  package.

1. SSPD (Symmetric Segment-Path Distance) [1]
2. OWD  (One-Way Distance) [2]
3. Hausdorff [3]
4. Frechet [4]
5. Discret Frechet [5]
6. DTW (Dynamic Time Warping) [6]
7. LCSS (Longuest Common Subsequence) [7]
8. ERP (Edit distance with Real Penalty) [8]
9. EDR (Edit Distance on Real sequence) [9]

* All distances but *Discret Frechet* and *Discret Frechet* are are available with *Euclidean* or *Spherical* option :
 *  *Euclidean* is based on Euclidean distance between 2D-coordinates.
 *  *Spherical* is based on Haversine distance between 2D-coordinates.

* Grid representation are used to compute the OWD distance. 

* Python implementation is also available in this depository but are not used within `traj_dist.distance` module.

## Dependencies

**trajectory_distance** is tested to work under Python 3.6 and the following dependencies:
 
* NumPy >= 1.16.2
* Cython >= 0.29.6
* shapely >= 1.6.4.post2
* geohash2 == 1.1
* pandas >= 0.24.2
* scipy >= 0.20.3
* A working C/C++ compiler.

## Install

This package can be build using `distutils`.

Move to the package directory and run :

```
python setup.py install 
```
to build Cython files. Then run:

```
pip install .
```
to install the package into your environment.

## How to use it

You only need to import the distance module.

```
import traj_dist.distance as tdist
```

All distances are in this module. There are also two extra functions 'cdist', and 'pdist' to compute pairwise distances between all trajectories in a list or two lists. 

Trajectory should be represented as nx2 numpy array. 
See `traj_dist/example.py` file for a small working exemple. 

Some distance requires extra-parameters.
See the help function for more information about how to use each distance.

## Performance

The time required to compute pairwise distance between 100 trajectories (4950 distances), composed from 3 to 20 points (`data/benchmark.csv`) :

| 		         | Euclidan      | Spherical |
| ------------- |:-------------:| -----:|
| discret frechet|0.0659620761871|-1.0|
|dtw | 0.0781569480896 | 0.114996194839|
|edr | 0.0695221424103 | 0.106939792633|
|erp | 0.171737909317 | 0.319380998611|
|frechet | 29.1885719299 | -1.0|
|hausdorff | 0.310199975967 | 0.780081987381|
|lcss | 0.0711951255798 | 0.111418008804|
|sowd grid, precision 5 | 0.164781093597 | 0.159924983978|
|sowd grid, precision 6 | 0.973792076111 | 0.954225063324|
|sowd grid, precision 7 | 7.62574410439 | 7.78553795815|
|sspd | 0.314118862152 | 0.807314872742|

See `traj_dist/benchmark.py` to generate this benchmark on your computer.

## References

1.  *P.  Besse,  B.  Guillouet,  J.-M.  Loubes,  and  R.  Francois,  “Review  and perspective   for   distance based trajectory clustering,”
arXiv preprint arXiv:1508.04904, 2015.*
2. *B. Lin and J. Su, “Shapes based trajectory queries for moving objects,”
in
Proceedings  of  the  13th  annual  ACM  international  workshop  on
Geographic information systems
.    ACM, 2005, pp. 21–30.*
3. *F. Hausdorff, “Grundz uge der mengenlehre,” 1914*
4. *H.  Alt  and  M.  Godau,  “Computing  the  frechet  distance  between  two
polygonal curves,”
International Journal of Computational Geometry &
Applications
, vol. 5, no. 01n02, pp. 75–91, 1995.*
5. *T. Eiter and H. Mannila, “Computing discrete fr
 ́
echet distance,” Citeseer,
Tech. Rep., 1994.*
6. *D. J. Berndt and J. Clifford , “Using dynamic time warping to find patterns in time series.” in KDD workshop, vol. 10, no. 16. Seattle, WA, 1994, pp. 359–370* 
7. *M. Vlachos, G. Kollios, and D. Gunopulos, “Discovering similar multi-
dimensional trajectories,” in
Data Engineering, 2002. Proceedings. 18th
International Conference on
.IEEE, 2002, pp. 673–684*
8. *L.  Chen  and  R.  Ng,  “On  the  marriage  of  lp-norms  and  edit  distance,”
in
Proceedings  of  the  Thirtieth  international  conference  on  Very  large
data bases-Volume 30
.    VLDB Endowment, 2004, pp. 792–803.*
9. *L. Chen, M. T.
 ̈
Ozsu, and V. Oria, “Robust and fast similarity search for
moving object trajectories,” in
Proceedings of the 2005 ACM SIGMOD
international  conference  on  Management  of  data
.      ACM,  2005,  pp.
491–502.*

