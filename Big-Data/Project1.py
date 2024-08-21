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

def hash_func(u, a, b, C, p):
	hashu = ((a*u+b)%p)%C
	return hashu

def triancount(pairs):
	result = CountTriangles(pairs[1])
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

def triancount2(pairs):
	result = CountTriangles(pairs)
	return [(0, result)]

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
	
def MR_ApproxTCwithSparkPartitions(edges, C):
	result = (edges.mapPartitions(triancount2)
	   .groupByKey()
	   .mapValues(lambda vals: (C ** 2) * sum(vals)))
	
	return result

def main():
	# CHECKING NUMBER OF CMD LINE PARAMTERS
	assert len(sys.argv) == 4, "Usage: python Project1.py <C> <R> <file_name>"

	# SPARK SETUP
	conf = SparkConf().setAppName('mysolve')
	sc = SparkContext(conf=conf)

	# INPUT READING

	# 1. Read number of partitions
	C = sys.argv[1]
	assert C.isdigit(), "C must be an integer"
	C = int(C)

	# 2. Read number of repetitions
	R = sys.argv[2]
	assert R.isdigit(), "R must be an integer"
	R = int(R)
	
	data_path = sys.argv[3]
	assert os.path.isfile(data_path), "File or folder not found"
	rawData = sc.textFile(data_path,minPartitions=C).cache()
	edges = rawData.map(lambda x: x.split(','))
	edges = edges.map(lambda x: (int(x[0]), int(x[1])))
	edges = edges.repartition(numPartitions=C)
	print("file name: " + str(data_path))
	print("Number of edges: " + str(edges.count()))
	print("C: " + str(C))
	print("R: " + str(R))
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
	print("Number of triangles by Spark partitions:")
	start = time()
	sp_part = MR_ApproxTCwithSparkPartitions(edges,C).collect()[0][1]
	end = time()
	print(sp_part)
	print((end-start)*1000)


if __name__ == "__main__":
	main()
