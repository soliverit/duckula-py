### Includes ###
## Native
from argparse			import ArgumentParser 
from os.path			import isfile
from sklearn.cluster	import KMeans
from pandas				import read_csv
## Project

##
# KMeans clustering: In case that's your thing. 
#
# A simple wrapper for KMeans in case we want to use it at some point.
##
class KMeansClusterer():
	CLUSTER_ID_COLUMN_NAME	= "cluster_id"
	##
	# Quick load: Spare the referencing script importing read_csv
	##
	@classmethod
	def QuickLoad(cls, path, labels):
		if isfile(path):
			return cls(read_csv(path), labels)
	##
	# params:
	#
	# data:			pandas.DataFrame
	# labels:		array string column labels
	# nCluster:		int number of clusters
	# randomState:	int random state seed
	##
	def __init__(self, data, labels, nClusters=4, randomState=1, maxIterations=50, nInit="auto", algorithm="lloyd"):
		self.data			= data			# DataFrame
		self.labels			= labels		# String[] of column labels
		self.nClusters		= nClusters		# int number of clusters
		self.randomState	= randomState	# int random seed
		self.maxIterations	= maxIterations	# int number of iterations during fitting
		self.nInit			= nInit			# string, callable,
		self.algorithm		= algorithm		# string algorithm name lloyd or elkan
		self.kmeans			= False			# KMeans placeholder
	##
	# Fit a model to self.data.
	#
	##
	def cluster(self):
		data									= self.data.copy()
		self.kmeans								= KMeans(
			n_clusters=		self.nClusters, 
			n_init=			self.nInit,
			algorithm=		self.algorithm,
			max_iter=		self.maxIterations,
			random_state=	self.randomState
		).fit(self.data)
	##
	# Get cluster IDs
	#
	# params:
	#	data:	pandas.DataFrame
	#
	# output:	np.array of cluster IDs
	##
	def predict(self, data):
			return self.kmeans.predict(data)