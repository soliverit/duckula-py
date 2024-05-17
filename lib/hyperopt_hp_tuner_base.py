### Includes ###
## Native 
from hyperopt.hp	import choice, uniform, uniformint
from hyperopt		import fmin, tpe, Trials
from numbers		import Number
from numpy			import average
## Project
class HyperoptHpTunerBase():
	def __init__(self, model, iterations=20, parameters={}, 
			  algorithm=tpe.suggest, trials=Trials(), cvSteps=1):
		self.model		= model				# Anything we can tune using setattr to modify params
		self.iterations	= iterations		# int number of stages in the tuning process
		self.parameters	= parameters.copy()	# Dict of hp.* parameter definitions
		self.algorithm	= algorithm			# Hyperopt tuning algorithm
		self.trials		= trials			# Hyperopt Trials result tracking object
		self.optimiser	= False				# Hyperopt fmin or similar
		self.cvSteps	= cvSteps			# int number of cross-validation steps
	##
	# Calcualte the model fitness (Intenral)
	#
	# Run the evaluate(Abstract) method self.cvSteps times and return the average score. Where
	# self.cvSteps defines cross-validation rounds.
	#
	# Process:
	#	- Apply parameters defined by hyperopt for the iteration
	#	- For self.cvSteps
	#	-- Score the model. Add to results set
	#	-- Apply intermediary model (see below on self.intermediaryModelChanges)
	#	- Return average
	#
	# Intermediary model changes self.intermediaryModelChanges (Virtual)
	#
	#	The intermediary model changes method is for apply cross-validation rules
	#	to the model. For example, the standard method here is shuffle split. To
	#	accommodate the shuffle split, we use self.intermediaryModelChanges to call
	# 
	##
	def _score(self, params):
		# Update the model
		for key, value in params.items():
			if hasattr(self.model, "updateHyperparameter"):
				self.model.updateHyperparameters(key, value)
			else:
				setattr(self.model, key, __class__.CastValueToExpceted(value))
		results = []
		# Do score with cross-validation
		for i in range(self.cvSteps):
			if i > 0:
				self.intermediaryModelChanges()
			results.append(self.evaluate())
		return average(results)
	##
	# Intermediary actions between cross-validation steps (Virtual)
	#
	# Do things to the data or whatever before the next step of cross validation. E.g, 
 	# shuffle / repartition data
	##
	def intermediaryModelChanges(self):
		pass
	##
	# Tune the model: Find the best hyperparameters (Final)
	##
	def tune(self):
		self.best = fmin(
			self._score, 
			self.parameters, 
			algo=tpe.suggest,
			max_evals=self.iterations, 
			trials=Trials()
		)
	##
	# Model fitness function (Abstract)
	#
	# This method should return a number denoting the fitness of the model.
	#
	#	output:	Number<any> indicating the quality of the model
	##
	def evaluate(self):
		raise Exception("%s doesn't override HyperoptHpTunerBase. evaluate method" %(__class__.__name__))
	############
	# Adding hyperparameters
	############
	def addParameter(self, code, parameter):
		self.parameters[code]	= parameter
	def addUniformParameter(self, code, min, max):
		self.parameters[code]	= uniform(code, min, max)
	def addUniformIntParameter(self, code, min, max):
		self.parameters[code]	= uniformint(code, min, max)
	def addChoiceParameter(self, code, options):
		self.parameters[code]	= choice(code, options)
	@staticmethod
	def CastValueToExpceted(value):
		return int(value) if isinstance(value, Number) and int(value) == value else value	
	@classmethod
	def MakeUniformParameter(self, code, min, max):
		return uniform(code, min, max)
	@classmethod
	def MakeUniformIntParameter(self, code, min, max):
		return uniformint(code, min, max)
	@classmethod 
	def MakeChoiceParameter(self, code, options):
		return choice(code, options)