### Include ###
## Native
from importlib				import import_module
from inspect				import getmembers, isclass, getmodule
from argparse				import ArgumentParser
from pandas					import read_csv
from sklearn.metrics 		import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing	import Normalizer, StandardScaler
## Project

##
# Estiamtor base: A class for interchangeable supervised learning models
##
class EstimatorBase():
	HYPEROPT_HP_TUNER_PARAMS	= {}
	@classmethod
	def QuickLoad(cls, path, target):
		return cls(read_csv(path), target)
	##
	# params:
	#	trainData:			DataFrame
	#	target:				string feature name (of column in trainData)
	#	trainTestSplit:		float 0 < x < 1 split of data for training and testing. x = train size
	##
	def __init__(self, trainData, target, trainTestSplit=0.5, customParams={}, 
			  scaler=StandardScaler(), normaliser=Normalizer()):
		self.data				= trainData			# DataFrame with features and target
		self.target				= target			# string target label
		self.trainTestSplit		= trainTestSplit	# float train/test split
		self.model				= False				# Trained model (XGBoost, MLP, or whatever)
		self.scaler				= scaler			# sklearn.preprocessing Scaler
		self.normaliser			= normaliser		# sklearn.preprocessing Normaliser
		self.applyScaler		= False				# bool apply scaler during preprocessing inputs
		self.applyNormaliser	= False				# bool apply normaliser during preprocessing inputs
		self.customParams		= customParams		# dict of params not covered by the default model
	##
	# Shuffle data
	##
	def shuffleData(self):
		self.data	= self.data.sample(frac=1).reset_index(drop=True)
	##
	# Get training data  with target
	#
	# output:	DataFrame of training data including targets, not test data
	##
	@property
	def trainingData(self):
		return self.data[: int(len(self.data) * self.trainTestSplit)]
	##
	# Get test data  with target
	#
	# output:	DataFrame of test data including targets, not training data
	##
	@property
	def testData(self):
		return self.data[int(len(self.data) * self.trainTestSplit) : ]
	##
	# Get training data  without targets + Preprocessing Standard/Normal etc
	#
	# output:	DataFrame of training data excluding targets, not test data
	##
	@property
	def trainingInputs(self):
		data	= self.data[: int(len(self.data) * self.trainTestSplit)]
		del data[self.target]
		return self.preprocessInputs(data)
	##
	# Get test data  without targets
	#
	# output:	DataFrame of test data excluding targets, not training data
	##
	@property
	def testInputs(self):
		data	= self.data[int(len(self.data) * self.trainTestSplit) : ]
		del data[self.target]
		return data
	##
	# Get training targets
	#
	# output:	array of training targets
	##
	@property
	def trainingTargets(self):
		return self.data[self.target].values[:  int(len(self.data) * self.trainTestSplit)]
	##
	# Get test targets
	#
	# output:	array of test targets
	##
	@property
	def testTargets(self):
		return self.data[self.target].values[ int(len(self.data) * self.trainTestSplit) : ]
	##
	# Preprocess input data (Virtual)
	#
	# Use this method to apply Scaler, Normalisers, whatever
	##
	def preprocessInputs(self, data):
		if self.applyScaler and self.scaler:
			data	= self.scaler.fit_transform(data)
		if self.applyNormaliser and self.normaliser:
			data	= self.normaliser.fit_transform(data)
		return data
	##
	# Preprocess targets (Virtual)
	##
	def preprocessTargets(self, targets):
		return targets
	##
	# Train model (Abstract)
	#
	# Train the model so you can make estimates
	##
	def train(self):
		raise "%s doesn't override abstract train method of EstimatorBase" %(__class__.__name__)
	##
	# Get model parameters (Abstract)
	# 
	# This method should produce an object that defines the model, typically a {}. The
	# config is used by train and printModelParams.
	#
	# For example, XGBoost might have {"n_rounds": 1000, "rate_drop": 0.1}
	# 
	# output:	{} or custom. Whatever's needed for the train method. Ideally, subscripted
	##
	@property
	def params(self):
		raise Exception("%s doesn't override abstract @property params method of EstimatorBase" %(__class__.__name__))
	##
	# Get all params: self.params + self.customParams
	##
	@property
	def allParams(self):
		params	= self.params
		for key in list(self.customParams):
			params[key]	= self.customParams[key]
		return params
	##
	# Run test using test data and return RMSE, R2, and MAE scores 
	#
	# output:	{"rmse": (root)mean_squared_error, "mae": <mean_absolute_error>, "r2": <r2_score>}
	##
	def test(self):
		if not self.model:
			self.train()
		predictions	= self.predict(__class__.DataFrameToInputType(self.preprocessInputs(self.testInputs)))
		return {
			"r2":	r2_score(predictions, self.testTargets),
			"rmse":	mean_squared_error(predictions, self.testTargets) ** 0.5,
			"mae":	mean_absolute_error(predictions, self.testTargets)
		}
	##
	# Predict target valeus from 2D feature array
	#
	# output:	Array of predictions/estimates
	##
	def predict(self, data):
		return self.model.predict(self.__class__.DataFrameToInputType(data))
	##
	# Get R2 without including it in every script uses an EstimatorBase object
	#
	# output:	float r2_score
	##
	def r2(self, data, targets):
		return r2_score(self.predict(data), targets)
	##
	# Get RMSE without including it in every script uses an EstimatorBase object
	#
	# output:	float mean_squared_error ^ 0.5
	##
	def rmse(self, data, targets):
		return mean_squared_error(self.predict(data), targets) ** 0.5
	##
	# Get mean absolute error without including it in every script uses an EstimatorBase object
	#
	# output:	float mean_absolute_error
	##
	def mae(self, data, targets):
		return mean_absolute_error(self.predict(data), targets)
	##
	# Convert DataFrame to native data type (Virtual)
	#
	# Convert a DataFrame to the data type expected by the model. This
	# mainly exists for XGBoost's DMatrix. Most don't need it.
	##
	@classmethod
	def DataFrameToInputType(cls, data):
		return data