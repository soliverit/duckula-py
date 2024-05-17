### Includes ###
## Native
from sklearn.ensemble	import GradientBoostingRegressor
from hyperopt.hp		import uniform, uniformint
## Project
from lib.estimator_base	import EstimatorBase
##
# Gradient-boosting Regressor, curtesy of sklearn.ensemble
#
# A wrapper for sklearn's GradientBoostingRegressor
##
class GBDTEstimator(EstimatorBase):
	##
	# Hyperparameters definitions for Hyperopt
	##
	HYPEROPT_HP_TUNER_PARAMS	= {
		"nEstimators": 		uniformint("nEstimators", 500, 2000),
		"learningRate":		uniform("learningRate", 0.01, 0.3),
		"minSampleSplit":	uniformint("minSampleSplit", 5,40),
		"minSamplesLeaf":	uniformint("minSamplesLeaf", 5, 40)
	}
	def __init__(self, data, target, trainTestSplit=0.5, nEstimators=100, learningRate=0.1,
			  minSamplesSplit=30, minSamplesLeaf=24):
		super().__init__(data, target, trainTestSplit=trainTestSplit)
		self.nEstimators		= nEstimators		# int number of estimators/rounds
		self.learningRate		= learningRate		# float step size
		self.minSamplesSplit	= minSamplesSplit	# int min  samples for a split
		self.minSamplesLeaf		= minSamplesLeaf	# int min samples for leafs
	##
	# Get model parameters (override Abstract)
	#
	# output:	Dict of mixed value parameters for the learner
	##
	@property
	def params(self):
		return {
			"learning_rate": 		self.learningRate,
			"n_estimators":			self.nEstimators,
			"min_samples_split":	self.minSamplesSplit,
			"min_samples_leaf":		self.minSamplesLeaf
		}
	##
	# Train the model
	##
	def train(self):
		self.model	= GradientBoostingRegressor(**self.allParams)
		self.model.fit(self.trainingInputs, self.trainingTargets)