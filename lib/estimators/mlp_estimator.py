### Includes ###
## Native
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.preprocessing	import Normalizer, StandardScaler
from hyperopt.hp			import choice, uniform, uniformint
### Project
from lib.estimator_base		import EstimatorBase
##
# MLP Estimator - Multilayer-perceptron regression, probably classification
#
# A wrapper for sklearn's MLP. Does regression, probably does classification
# but I haven't tried that, yet.
##
class MLPEstimator(EstimatorBase):
	##
	# Hyperparameters definitions for Hyperopt
	##
	HYPEROPT_HP_TUNER_PARAMS	= {
		"maxIterations": 	uniformint("nEstimators", 500, 2000),
		"alpha":			uniform("alpha", 0.01, 0.3),
		"solver":			choice("solver", ["adam", "lbfgs", "sgd"]),
		"layers":			choice("layers", [(64, ), (100, 64,), (50, 50, 100,)]),
	}
	##
	# params:
	#	data:			DataFrame containing features and targets for training and test data
	#	target:			string target column name
	#	trainTestSplit:	float train/test split 0 < x  < 1 ( or 1 if you're not going to use test())
	#	scaler:			sklearn.preprocessing Scale. only used if self.applyScaler == True
	#	normaliser:		sklearn.preprocessing Normaliser. only used if self.applyNormaliser == True
	#	maxIterations:	int max iterations (not epochs). 
	#	randomState:	int Random seed
	#	solver:			string solver: adam, lfbgs, sgd, etc
	#	alpha:			float regularisation coefficient
	#	layers:			Tuple[int] number of hidden layers, per layer. E.g (50, 20,) for 2 layers with 50 and 20 neurons, respectively
	#	customParams:	dict of anything that MLPRegressor/MLPClassifier will accept as a labeled parameter akin to "max_iter"
	##
	def __init__(self, data, target, trainTestSplit=0.5, scaler=StandardScaler(), normaliser=Normalizer(),
			  maxIterations=1000, randomState=1, solver="adam", alpha=0.005, layers=False, customParams={},
			  mlpType=MLPRegressor):
		super().__init__(data, target, trainTestSplit=trainTestSplit, scaler=scaler, normaliser=normaliser, customParams=customParams)
		self.maxIterations		= maxIterations	# int max iterations
		self.randomState		= randomState	# int random seed
		self.solver				= solver		# string solver. E.g adam, LBFGS, or stochastic gradient descent
		self.alpha				= alpha			# float regularisation coefficient
		self.mlpType			= mlpType		# MLPRegressor or MLPClassifer
		# Assume 2x features, single layer unless explictly defined
		self.layers	= layers if layers else (len(data.columns) * 2 - 2,)
		# Tell self.preprocessInputs to apply the scaler and normaliser
		self.applyScaler		= True		# bool apply scaler during preprocessing inputs
		self.applyNormaliser	= True		# bool apply normaliser during preprocessing inputs
	##
	# Get params: (override Abstract)
	#
	# output:	dict of parameters defined by instance members
	##
	@property
	def params(self):
		return {
			"hidden_layer_sizes":	self.layers,
			"max_iter":				self.maxIterations,
			"solver":				self.solver,
		}
	##
	# Train model (override Abstract)
	##
	def train(self):
		data	= self.trainingInputs
		# Fit and apply scale and normaliser if they exist
		if self.applyScaler:
			data	= self.scaler.fit_transform(data)
		if self.applyNormaliser:
			data	= self.normaliser.fit_transform(data)
		# Prepare the model
		self.model	= self.mlpType(**self.allParams)
		# Fit the model
		self.model.fit(data, self.trainingTargets)