from igraph import *
from math import *
import sys
import numpy as np
import os
import glob
from random import shuffle
import matplotlib.pyplot as plt

#######################################################
############### HELPER FUNCTIONS ######################
#######################################################

# Returns membership (random assignment) of vertices into g groups/partitions
def get_membership(n,g):
	membership = [i%g for i in range(n)]
	shuffle(membership)
	return membership

# Returns n X 1 vector ei of the kth partition
def membership_of(membership,k):
	return [1 if x==k else 0 for x in membership]

# Returns n X n diagonal degree matrix of graph
def get_degree_matrix(graph,n):
	return [[graph.degree(i) if i==j else 0 for i in range(n)] for j in range(n)]

# Returns Root Euclidean distance as defined in paper
def rooted(S1,S2,n,g):
	return sqrt(sum([sum([(sqrt(S1[i][j]) - sqrt(S2[i][j]))**2 for j in range(g)]) for i in range(n)]))

# Returns moving range average, M
def moving_range_average(ts):
	return sum([abs(ts[i]-ts[i-1]) for i in range(1,len(ts))])/(len(ts)-1)

def writeToFile(pairwise_similarity,file_prefix):
	file_name = file_prefix + "_time_series.txt"
	fo = open(file_name, "w+")
	for sim in pairwise_similarity:
		fo.write(str(sim))
		fo.write("\n")
	fo.close()
	
# Transforms the vertices in Graph G1 and G2 to zero-based vertex indexing for performance optimization
def transform_edgelists(edge_list1,edge_list2):
	mymap = {}
	i = 0
	for x,y in edge_list1+edge_list2:
		if x not in mymap:
			mymap[x] = i
			i = i + 1
		if y not in mymap:
			mymap[y] = i
			i = i + 1
	new_edge_list1 = map(lambda x: (mymap[x[0]],mymap[x[1]]) ,edge_list1)
	new_edge_list2 = map(lambda x: (mymap[x[0]],mymap[x[1]]) ,edge_list2)
	return new_edge_list1,new_edge_list2


#######################################################
############### DELTACON Algorithm 2 ##################
#######################################################

def deltacon_sim(edge_list1,edge_list2,g):
	edge_list1,edge_list2 = transform_edgelists(edge_list1,edge_list2)
	graph1 = Graph(edge_list1)
	graph2 = Graph(edge_list2)

	# Since we need to take Union of vertices in graph1 and graph2, we reinitialize the graph with n
	n = max(graph1.vcount(),graph2.vcount())
	graph1 = Graph(n,edge_list1)
	graph2 = Graph(n,edge_list2)

	g = min(g,n) # It should be that g << n , but incase it is not then we take n

	A1 = np.array(list(graph1.get_adjacency()))
	A2 = np.array(list(graph2.get_adjacency()))
	I = np.identity(n)
	D1 = np.array(get_degree_matrix(graph1,n))
	D2 = np.array(get_degree_matrix(graph2,n))

	eps1= 1/float(1 + max(D1[i][i] for i in range(len(D1))))
	eps2= 1/float(1 + max(D2[i][i] for i in range(len(D2))))

	membership = get_membership(n,g)	
	S1 = []
	S2 = []
	for k in range(g):
		s0k = membership_of(membership,k)
		S1.append(np.dot(np.linalg.inv(I + eps1**2*D1 - eps1*A1),s0k))
		S2.append(np.dot(np.linalg.inv(I + eps2**2*D2 - eps2*A2),s0k))
	
	S1 = np.array(S1).transpose().tolist()
	S2 = np.array(S2).transpose().tolist()
	return 1/float(1 + rooted(S1,S2,n,g))


def main():
	if len(sys.argv) < 2 :
		print "python anomaly.py dataset/dataset/voices/"
		exit(1)
	path = sys.argv[1].strip()
	dataset = path.split('/')[-2]
	number_of_files = len(glob.glob(path+'[0-9]*_'+dataset+'.txt'))
	
	edge_lists = [0]*number_of_files
	for i in range(number_of_files):
		file_name = path+str(i)+"_"+dataset+".txt"
		edge_list_file = open(file_name)
		edge_lists[i] = map(lambda x:tuple(map(int,x.split())),edge_list_file.read().split("\n")[:-1]) 
		edge_list_file.close()

	g = 40 # No of groups/partitions

	# Generating pairwise similarity list using the deltacon_sim function
	pairwise_similarity = []
	for i in range(number_of_files-1):
		pairwise_similarity.append(deltacon_sim(edge_lists[i],edge_lists[i+1],g))
		
	# Printing pairwise_similarity
	print "Similarity Scores :"
	for i,x in enumerate(pairwise_similarity):
		print "Similarity(G"+str(i)+",G"+str(i+1)+") = "+str(x)

	# Writing similarity to file
	writeToFile(pairwise_similarity,dataset)

	# Calculating the threshold value for determining anomalies
	M = moving_range_average(pairwise_similarity)
	median = np.median(np.array(pairwise_similarity))

	# Threshold chosen as median +/- M
	lower_threshold = median - M
	upper_threshold = median + M

	# Using only the lower bound threshold for getting anomaly
	print "\nAnamolous Graphs are :"
	for i in range(len(pairwise_similarity)-1):
		if pairwise_similarity[i] < lower_threshold and pairwise_similarity[i+1] < lower_threshold:
			print "G"+str(i+1)

	# Plotting
	plt.plot(pairwise_similarity)
	plt.ylabel('Pairwise Similarity')
	plt.xlabel('Time Points')
	lower_threshold_line = plt.axhline(y = lower_threshold,xmin=0,xmax=1,hold= None,color = "red",label = "Lower Threshold")
	median_line = plt.axhline(y = median ,xmin=0,xmax=1,hold= None,color = "green",label = "Median")
	upper_threshold_line = plt.axhline(y = upper_threshold,xmin=0,xmax=1,hold= None,color = "gray",label = "Upper Threshold")
	plt.legend([upper_threshold_line, median_line, lower_threshold_line], ["Upper Threshold", "Median","Lower Threshold"])
	plt.show()
	

if __name__ == "__main__":
	main()