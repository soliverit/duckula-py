### Includes ###
## Native 
from xgboost				import DMatrix, train
from hyperopt.hp			import uniform, uniformint
## Project
from lib.estimator_base		import EstimatorBase
##
# XGBoost estimator: An estimator using XGBoost 2
##
class XGBoostEstimator(EstimatorBase):
	HYPEROPT_HP_TUNER_PARAMS	= {
		"learningRate":	uniform("learningRate", 0.03, 0.3),
		"rateDrop":		uniform("rateDrop", 0.01, 0.2),
		"skipDrop":		uniform("skipDrop", 0.3, 0.7),
		"maxDepth":		uniformint("maxDepth", 2, 10),
		"nRounds":		uniformint("nRounds", 700, 1200)
	}
	def __init__(self, trainData, target, trainTestSplit=0.5, booster="dart", maxDepth=6,
			  learningRate=0.1, objective="reg:squarederror", sampleType="uniform",
			  normaliseType="tree", rateDrop=0.1, skipDrop=0.5, nRounds=100, gamma=0):
		super().__init__(trainData, target, trainTestSplit=trainTestSplit)
		self.booster		= booster		# string Boosting algorithm
		self.maxDepth		= maxDepth		# int Max branches
		self.learningRate	= learningRate	# float Learning rate: solution space step size or something to that effect 
		self.objective		= objective		# string determines regression or classification and score metric.
		self.sampleType		= sampleType	# string sampling method. xgboost constructor property
		self.normaliseType	= normaliseType	# string normalisation method. xgboost constructor property
		self.rateDrop		= rateDrop		# float 0 < x < 1 probability that a learner are dropped during an iteration. xgboost constructor property
		self.skipDrop		= skipDrop		# float 0 < x < 1 probability rate drop will be ignored
		self.nRounds		= nRounds		# int No. of rounds. Basically No. estimators from Random forest or GBDT
		self.gamma			= gamma			# float min loss reduction required before further paritioning a leaf
	##
	# Get model parameters (override Abstract)
	#
	# output:	Dict of mixed value parameters for the learner
	##
	@property
	def params(self):
		params =  {
			"booster": 			self.booster,
			"max_depth": 		self.maxDepth, 
			"learning_rate":	self.learningRate,
			"objective": 		self.objective,
			"sample_type": 		self.sampleType,
			"normalize_type": 	self.normaliseType,
			"rate_drop": 		self.rateDrop,
			"skip_drop": 		self.skipDrop,
			"gamma":			self.gamma
		}
		if self.booster == "dart":
			params["skip_drop"]	= self.skipDrop
		return params
	##
	# Train the model (Abstract)
	##
	def train(self):
		# WARNING!!! Always do params first! They change inline configs like train_test_split
		# TODO: Make params use underscore so we don't have to bridge it awkwardly
		params		= self.allParams
		data 		= DMatrix(self.trainingInputs, self.trainingTargets)
		self.model 	= train(params, data, self.nRounds)
	##
	# Extra config params for PrintModelConfig (Virtual)
	##
	def extraConfigParams(self):
		return {
			"nRounds":	self.nRounds
		}
	##
	# Convert DataFrame to native data type (Virtual)
	##
	@classmethod
	def DataFrameToInputType(cls, data):
		return DMatrix(data)
	