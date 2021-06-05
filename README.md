## GraphCore

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
$ p pathGeneration.py -filepath sample/0.txt -L 3 -N 3
INFO:root:['A', 'C', 'D']
INFO:root:['D', 'C', 'D']
INFO:root:['D', 'C', 'D']
```