# Big-Data
Big Data projects

In these projects was used Spark (Pyspark). 

Files facebook_large and facebook_small can be used to obtain and evaluate results of different algorithms used to calculate number of triangles. Project1 and Project2 use different approaches to calculate number of triangles.

Usage Project1: python Project1.py <C> <R> <file_name>
C - number of partitions
R - number of repetitions
file_name - one of them: facebook_large, facebook_small

Usage Project2: python Project2.py <C> <R> <F> <file_name>"
C - number of partitions
R - number of repetitions
F - 0 or 1, 0 for MR_ApproxTCwithNodeColors and 1 for MR_ExactTC
file_name - one of them: facebook_large, facebook_small
