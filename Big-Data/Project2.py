from pyspark import SparkContext, SparkConf
import numpy as np
import sys
import os
import random as rand
from time import time
from collections import defaultdict

def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if v > u:
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if w > v and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count

def countTriangles2(colors_tuple, edges, rand_a, rand_b, p, num_colors):
    #We assume colors_tuple to be already sorted by increasing colors. Just transform in a list for simplicity
    colors = list(colors_tuple)  
    #Create a dictionary for adjacency list
    neighbors = defaultdict(set)
    #Creare a dictionary for storing node colors
    node_colors = dict()
    for edge in edges:

        u, v = edge
        node_colors[u]= ((rand_a*u+rand_b)%p)%num_colors
        node_colors[v]= ((rand_a*v+rand_b)%p)%num_colors
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph
    for v in neighbors:
        # Iterate over each pair of neighbors of v
        for u in neighbors[v]:
            if u > v:
                for w in neighbors[u]:
                    # If w is also a neighbor of v, then we have a triangle
                    if w > u and w in neighbors[v]:
                        # Sort colors by increasing values
                        triangle_colors = sorted((node_colors[u], node_colors[v], node_colors[w]))
                        # If triangle has the right colors, count it.
                        if colors==triangle_colors:
                            triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count



def hash_func(u, a, b, C, p):
	hashu = ((a*u+b)%p)%C
	return hashu

def triancount(pairs):
	result = CountTriangles(pairs[1])
	return [(0, result)]

def triancount2(colors_tuple, edges, rand_a, rand_b, p, num_colors):
	result = countTriangles2(colors_tuple, edges, rand_a, rand_b, p, num_colors)
	return [(0, result)]

def colors(x,a, b, C, p):
	u = x[0]
	v = x[1]
	hashu = hash_func(u, a, b, C, p)
	hashv = hash_func(v, a, b, C, p)

	if hashu==hashv:
		return [(hashu,(u,v))]
	else:
		return []

def colors_exact(x, a, b, C, p):
	u = x[0]
	v = x[1]
	pairs = []
	hashu = hash_func(u, a, b, C, p)
	hashv = hash_func(v, a, b, C, p)
	for i in range(C):
		k_i = tuple(sorted([hashu,hashv,i]))
		pairs.append((k_i,(u,v)))
	return pairs


def MR_ApproxTCwithNodeColors(edges, C):
	p = 8191
	a = rand.randint(1,8190)
	b = rand.randint(0,8190)
	
	result = (edges.flatMap(lambda x: colors(x,a, b, C, p))
			.groupByKey()
			.flatMap(triancount)
			.groupByKey()
			.mapValues(lambda vals: (C ** 2) * sum(vals)))
	return result
	
def MR_ExactTC(edges, C):
	p = 8191
	a = rand.randint(1,8190)
	b = rand.randint(0,8190)

	result = (edges.flatMap(lambda x: colors_exact(x,a, b, C, p))
			.groupByKey()
			.flatMap(lambda y: triancount2(y[0], y[1], a, b, p, C))
			.groupByKey()
			.mapValues(lambda vals: sum(vals)))
	
	return result

def main():
	# CHECKING NUMBER OF CMD LINE PARAMTERS
	assert len(sys.argv) == 5, "Usage: python Project2.py <C> <R> <F> <file_name>"

	# SPARK SETUP
	conf = SparkConf().setAppName('mysolve')
	sc = SparkContext(conf=conf)

	# INPUT READING

	# 1. Read number of partitions
	C = sys.argv[1]
	assert C.isdigit(), "C must be an integer"
	C = int(C)
	print(C)
	# 2. Read number of repetitions
	R = sys.argv[2]
	assert R.isdigit(), "R must be an integer"
	R = int(R)
	F = sys.argv[3]
	assert F.isdigit(), "F must be an integer"
	F = int(F)
	
	data_path = sys.argv[4]
	
	rawData = sc.textFile(data_path,minPartitions=C).cache()
	edges = rawData.map(lambda x: x.split(','))
	edges = edges.map(lambda x: (int(x[0]), int(x[1])))
	edges = edges.repartition(numPartitions=32)
	print("file name: " + str(data_path))
	print("Number of edges: " + str(edges.count()))
	print("C: " + str(C))
	print("R: " + str(R))
	print("F: " + str(F))

	if F == 0:
		times = []
		results = []
		for i in range(R):
			start = time()
			result = MR_ApproxTCwithNodeColors(edges, C).collect()[0][1]
			results.append(result)
			end = time()
			times.append((end-start)*1000)
		print("Number of triangles by Node Colors:")
		print("Median: " + str(np.median(np.array(results))))
		print("Average time: " + str(np.mean(np.array(times))))
	
	else:
		times = []
		for i in range(R):
			start = time()
			exact_tc = MR_ExactTC(edges,C).collect()[0][1]
			end = time()
			times.append((end-start)*1000)
		print("Number of exact triangles:" + str(exact_tc))
		print("Average time: " + str(np.mean(np.array(times))))


if __name__ == "__main__":
	main()