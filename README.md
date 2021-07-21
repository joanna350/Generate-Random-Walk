### Generate Random Walk
- given a graph in the format of the files under the folder sample/, returns possible random walks
- Tarjan algorithm utilised for efficiency over existing modules

#### Environment
```
Python 3.9.5
macOS Catalina 10.15.7
```

#### Dependencies

```
random
pathlib
logging=='0.5.1.2'
argparse=='1.1'
networkx=='2.5.1'
```

#### Directory

- sample/*.txt : Folder with graph data following the format
- pathGeneration.py : Main file to invoke with arguments

#### To run on command line

```
$ python3 pathGeneration.py -filepath sample/0.txt -L 3 -N 3
INFO:root:['A', 'C', 'D']
INFO:root:['D', 'C', 'D']
INFO:root:['D', 'C', 'D']
```
