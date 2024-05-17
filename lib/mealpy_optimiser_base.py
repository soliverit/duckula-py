### Includes ###
## Native 
from typing 	import Any
from mealpy		import IntegerVar, FloatVar
from importlib	import import_module
from pkgutil	import walk_packages
from argparse	import ArgumentParser
class MealPyOptimiserBase():
	CONSTRCUTORS = {}
	FLOAT_VAR	= FloatVar
	INTEGER_VAR	= IntegerVar
	def __init__(self, data=False, epochs=100, minMax="min", varType=IntegerVar, algorithm=False, 
			  customParams={}, inequality=-1, population=50, logPath=None):
		self.data			= data				# DatasetBase 
		self.epochs			= epochs			# int number of epoch	
		self.minMax			= minMax			# string [min]imise or [max]imise
		self.varType		= varType			# MealPy variable type. Typically IntegerVar or FloatVar
		self.logPath		= logPath			# string path for log file
		self.lastResult		= False				# Agent or whatever comes from the solver
		self.inequality		= inequality		# int/float penalty objective
		self.population		= population		# int population size
		self.customParams	= customParams		# Dict of extra problem parameters. E.g obj_weights for multi-objective
		self.lastResult		= False				# Last result from optimisation
		self.algorithm		= algorithm			# The MealPy "magic" alogrithm of your choice. MUST DEFINE
	##
	# Lower bound of all variables (Virtual)
	##
	@property
	def lowerBounds(self):
		return [0.0 for i in range(self.data.length)]
	##
	# Upper bound of all variables (Abstract)
	##
	@property
	def upperBounds(self):
		raise Exception("%s doesn't override upperBounds property" %(__class__.__name__))
	##
	# Problem core definition: (Virtual but should return the current values)
	##
	@property
	def problem(self):
		return {
			"obj_func": self.score,
			"bounds":	self.varType(lb=self.lowerBounds, ub=self.upperBounds),
			"minmax":	self.minMax,
			"log_to":	self.logPath
		}
	##
	# Merge default problem and customParam dictionaries (Final)
	##
	@property
	def completeProblem(self):
		problem	= self.problem
		for key, value in self.customParams.items():
			problem[key]	= value
		return problem
	##
	# Score / fitness function (Abstract)
	##
	def score(self, solution):
		raise Exception("%s doesn't override score()" %(__class__.__name__))
	##
	# Solve the problem (Final [ideally])
	##
	def solve(self):
		self.solver			= self.algorithm(epoch=self.epochs, pop_size=self.population)
		self.lastResult 	= self.solver.solve(self.completeProblem)
	#####################
	### Magic methods ###
	#####################
	##
	# Barrage: Try all the models
	#
	# output: MealPy metaheuristic
	#
	## 
	def barrage(self, models=False):
		print("WARNING! This dumps results for effect. Doesn't affect anything, just looks cool") 
		from colorama	import init, Fore, Style
		init()
		__class__.MapConstructors()
		if not models:
			models	= list(__class__.CONSTRCUTORS)
		results		= []
		best		= 99999999999
		bestModel	= False
		for model in models:
			self.algorithm	= __class__.CONSTRCUTORS[model]
			while len(model) < 18:
				model += " "
			try:
				self.solve()
				result	= self.lastResult.target.objectives[0]
				results[model]	= result
				if result <= best:		
					if result == best:
						print("%s%s: %s%s" %(Fore.YELLOW, model, result, Style.RESET_ALL))
					else:
						print("%s%s: %s%s" %(Fore.GREEN, model, result, Style.RESET_ALL))
					best 		= result
					bestModel	= model
				else:
					print("%s: %s" %(model, result))
			except:
				print("%s: No solution " %(model))
		return bestModel
	################################################
	# Class and static stuff
	################################################
	@staticmethod
	def MapConstructors():
		# Only load once
		if not __class__.CONSTRCUTORS:
			# Dynamically import the target package
			package = import_module("mealpy")
			# Walk through all modules and submodules in the package
			for _, module_name, _ in walk_packages(package.__path__, "mealpy" + '.'):
				# Import the module
				module = import_module(module_name)
				# Inspect the module for classes
				for attribute_name in dir(module):
					attribute = getattr(module, attribute_name)
					if isinstance(attribute, type) and "_based" in str(attribute.__module__):
						__class__.CONSTRCUTORS[attribute.__name__] = attribute
			# Remvoe QTable and broken algorithms
			del __class__.CONSTRCUTORS["WMQIMRFO"]		# Never worked with anything I've tested
			del __class__.CONSTRCUTORS["OriginalICA"]	# Suuuuuppppppeeerrr slow or doesn't converge. Either way
			del __class__.CONSTRCUTORS["QTable"]		# Not an algorithm
		