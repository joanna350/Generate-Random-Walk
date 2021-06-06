from random import choice
import pathlib
import logging
import argparse
import time


def argParser():
    '''
    utility function to receive the file path to read from, length and number of paths to return
    as arguments in command line
    '''
    parser = argparse.ArgumentParser(description='pass the filepath, length and number of paths to return')
    parser.add_argument('-filepath', action='store', type=pathlib.Path)
    parser.add_argument('-L', action='store', type=int)
    parser.add_argument('-N', action='store', type=int)

    args = vars(parser.parse_args())

    return args['filepath'], args['L'], args['N']


def binS(arr, L):
    '''
    utility function for step 1 of generatePath
    :param List[int] arr: given an array
    :param int L: and the number with which to compare
    :return int ptr: from which the sorted array will be greater than or equal to L
    '''
    lptr, rptr = 0, len(arr) - 1
    while lptr < rptr:
        mid = lptr + (rptr - lptr) // 2
        if arr[mid] < L:
            lptr = mid + 1
        elif arr[mid] > L:
            rptr = mid
        else:
            return mid
    return lptr


class Graph:
    '''
    builds a graph based on the given format of graph
    by faithfully following the definition of random walk, generates paths that meet the criteria
    '''
    def __init__(self, fp, L, N):
        '''
        constructor
        :param PosixPath fp: filepath to process for graph creation
        :param int L: length of path in search (threshold for depth)
        :param int N: number of path to generate
        '''
        self.L = L
        self.N = N
        # prepare datastructures
        self.all, self.leaves, self.radj, self.adj = self.processData(fp)
        # for dfs tree to identify strongly connected components (SCC)
        self.V = len(self.all)
        self.Time = 0
        # to retrieve paths that meet the given conditions
        self.maxD = {} # (k:v) = (node: max_d)
        self.cycle = set()
        # primary initialization of depth per node
        for n in self.leaves:
            self.maxD[n] = 1

    def processData(self, fp):
        '''
        parse the data and return the datas tructures necessary for further work
        :return set() all: set of all nodes
        :return set() leaves: set of outdegree 0 nodes
        :return {Str: List[Str]} adj: (outgoing node : [incoming node])
        :return {Str: List[Str]} radj: exact opposite of adj
        :return List[List[Str, Str]] _edges: compilation of edges
        '''
        # read in the file that expresses graph in the format mentioned
        file = open(fp, 'r').read()
        edges = file.split('\n')

        # initialise placeholder
        adj, radj = dict(), dict()
        ind, outd = set(), set()
        for e in edges:
            o, i = e.split(' -> ')
            if o not in adj:
                adj[o] = [i]
            else:
                adj[o].append(i)
            if i not in radj:
                radj[i] = [o]
            else:
                radj[i].append(o)
            ind.add(i)
            outd.add(o)

        # Find the set of nodes with no outgoing degree
        all = ind.union(outd)
        leaves = all - outd
        return all, leaves, radj, adj

    def generatePath(self):
        '''
        0. prepare max depth per node based on the features of directed graph
           > leaves depth = 1, cycles depth = inf (updated in constructor)
           > nodes absorbing into cycles = inf
           > nodes absorbing into leaves (only) can be updated reverse, incrementally
        1. start at nodes with max depth >= L by uniform distribution
        2. choose next node at each step by uniform distribution
        :return List[Str] paths: that meet the criteria L / N, ', ' btw nodes
        '''

        # step 0
        # nodes that absorb into cycle will be part of inf
        self.updateInf()
        # ready to calculate depth in reverse from the leaves
        self.updateMaxD()

        # step 1
        orderbyD, ptr = self.randomWalk()

        # step 2
        paths = [] # mutable object will be updated after a call
        self.randomWalkUtil(paths, orderbyD, ptr)

        return paths

    def randomWalk(self):
        '''
        utility function to define the set of starting nodes
        :return List[[int, Str] orderbyD: List that holds (depth, node) information
        :return int ptr: pointer/index for the sorted list from where the value >= desired L
        '''
        orderbyD = []
        for n, d in self.maxD.items():
            orderbyD.append([d, n])
        # retain the node information to use later
        orderbyD.sort(key=lambda x: x[0])

        # (cleaning to be done later) the corner case
        # when demanded of higher length than possible from a given graph
        if self.L > orderbyD[-1][0]:
            msg = 'Choose L <= ' + str(orderbyD[-1][0])
            return [msg]

        # same order with the List[List[int, Str]] that holds (depth, node)
        possibleD = [x[0] for x in orderbyD]

        # binary search to find the point where depth >= L
        ptr = binS(possibleD, self.L)

        return orderbyD, ptr

    def randomWalkUtil(self, paths, orderbyD, ptr):
        '''
        utiltiy function to find the paths by choosing a node at each step
        from a set that has the potential to meet the path length
        and continue onward with the information of adjacent nodes
        until all the number of paths with requested length are generated
        '''
        while len(paths) < self.N:
            # initialize placeholders
            path = []
            startW = orderbyD[ptr:]
            l = self.L
            while len(path) < self.L:
                # uniformly distributed probabilistic choice of the next node
                nexp = choice(startW)
                path.append(nexp[1])
                l -= 1
                if nexp[1] not in self.leaves:
                    startW = []  # initialize for next
                    for nei in self.adj[nexp[1]]:
                        if self.maxD[nei] >= l: # filter for comp efficiency
                            startW.append([self.maxD[nei], nei])
                # reached leaves before meeting the length
                elif len(path) < self.L:
                    break # rerun
                if len(path) == self.L:
                    paths.append(path)

    def updateMaxD(self):
        '''
        utility function to update incrementally the nodes that leads to the leaves (sink) only
        similar to the updateInf, with less conditional check
        '''
        newVisit = self.leaves.copy()
        while newVisit:
            toVisit = newVisit.copy()
            newVisit = set()
            for n in toVisit: # iterate through node from a chosen set
                if n in self.radj: # if these nodes have incoming nodes
                    for node in self.radj[n]: # iterate through the incoming nodes
                        # if they are yet to be updated or can have higher value from the node n
                        if (node not in self.maxD) or (self.maxD[node] < self.maxD[n]+1):
                            self.maxD[node] = self.maxD[n]+1 # update based on current node n
                            newVisit = newVisit.union(self.radj[n]) # update the nodes to visit

    def updateInf(self):
        '''
        utility function to update the nodes that lead to cycles
        which indicate their potential to be marked as infinite for maximum depth
        '''
        newVisit = self.cycle.copy()
        while newVisit:
            toVisit = newVisit.copy()
            newVisit = set()
            for n in toVisit: # per each node from the chosen set
                if n in self.radj: # if there is a node incoming
                    for node in self.radj[n]: #  iterate through the incoming nodes
                        if node in self.cycle: # if it is in cycle (that we started with)
                            # then add all its neighboring nodes as well, excluding cycle
                            newVisit = newVisit.union(set(self.radj[node]) - self.cycle)
                        # if not the case nor updated, update and add to the set to visit
                        elif (node not in self.maxD) or self.maxD[node] < float('inf'):
                            self.maxD[node] = self.maxD[n]+1 # update based on prev node
                            newVisit.add(node)

    def SCCUtil(self, u, minToD, discovered, stackM, stack):
        '''
        utility function for detecting scc
        intermediary datastructures for recursive call are as follows
        :param Str u: node yet to be discovered
        :param dict() minToD: minimum time to discovery
        :param dict() discovered: primary discovery time
        :param dict() stackM: stack membership
        :param List stack: datastructure to check each subtree connection with the current node as root
        '''
        discovered[u] = self.Time
        minToD[u] = self.Time
        self.Time += 1
        stackM[u] = True
        stack.append(u)
        if u in self.adj:
            for v in self.adj[u]:
                if discovered[v] == -1:
                    self.SCCUtil(v, minToD, discovered, stackM, stack)
                    minToD[u] = min(minToD[u], minToD[v])
                elif stackM[v] == True:
                    minToD[u] = min(minToD[u], discovered[v])
        w = -1
        if minToD[u] == discovered[u]:
            cc = []
            while w != u:
                w = stack.pop()
                cc.append(w)
                stackM[w] = False
            if len(cc) >  1:
                for n in cc:
                    self.maxD[n] = float('inf')
                self.cycle = self.cycle.union(set(cc))
    # dfs tree
    def SCC(self):
        '''
        Tarjan algorithm
        detects nodes in cycle, update self.cycle and self.maxD accordingly
        '''
        disc = {}
        low = {}
        stackM ={}
        for n in self.all:
            disc[n] = -1
            low[n] = -1
            stackM[n] = -1
        stack = []
        for n in self.all:
            if disc[n] == -1:
                self.SCCUtil(n, low, disc, stackM, stack)

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    fp, L, N = argParser()
    start = time.time()
    g = Graph(fp, L, N)
    g.SCC()
    output = g.generatePath()
    end = time.time()
    print('time taken: {}'.format(end -start))
    # format
    for o in output:
        logging.info(o)