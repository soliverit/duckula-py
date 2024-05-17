### Includes ###
## Native
from sklearn.svm	import SVR
from hyperopt.hp	import uniform
## Project
from lib.estimator_base	import EstimatorBase

class SVREstimator(EstimatorBase):
	HYPEROPT_HP_TUNER_PARAMS = {
		"C": 		uniform("C", 0, 50),
		"epsilon":	uniform("epsilon", 0.001, 0.5)
	}
	def __init__(self, data, target, trainTestSplit=0.5, customParams={}, C=1, epsilon=0.1):
		super().__init__(data, target, trainTestSplit=trainTestSplit, customParams=customParams)
		self.C					= C
		self.epsilon			= epsilon
		self.applyScaler		= True
		self.applyNormaliser	= True
	@property
	def params(self):
		return {"C": self.C, "epsilon": self.epsilon}
	def train(self):
		self.model	= SVR(**self.allParams)
		self.model.fit(self.trainingInputs, self.trainingTargets)
