# Duckula DS - Data science tools for dabblers
A library for data scientists with no data science or programming experience. Easily create supervised machine learning models, solve optimisation problems, and tune the hyperparameters. 

- Supervised machine learning model creation and tuning
- Optimisation with 193 optimisation algorithms implemented by MealPy
- Hyperparameter tuning with Hyperopt
- KMeans clustering

# Getting started
## Prerequisites
- Get Python `3.x` - Developed with `3.11.4`
- Install necessary packages `pip install -r requirements.txt`
## Solving optimisation problems
Overview video [here](https://www.youtube.com/watch?v=Dzri5ZaPAIk): 
### Using a manually selected MealPy algorithm
````Python
# Import MealPy metaheuristics library wrapper
from lib.mealpy_optimiser_base	import MealPyOptimiserBase
# Load MealPy algorithm dict
MealPyOptimiserBase.MapConstructors()
###
# Let's solve the Rosenbrock problem: f(x, y) = (x - a) ^ 2 + b(y - x ^ 2) ^ 2
###
class Optimiser(MealPyOptimiserBase):
	## We need to define the boundaries with @property methods for variably bounds
	@property
	def lowerBounds(self):
		return [-5, -5]
	@property
	def upperBounds(self):
		return [5, 5]
	## We need to define a fitness function
	def score(self, solution):
		a, b	= 1, 100
		x, y	= solution[0], solution[1]
		return (x - a) ** 2 + b * (y - x**2) ** 2
# Create instance with manually chosen MealPy algorithm. All stored in wrapper's CONSTRUCTORs dict
optimiser	= Optimiser(
	algorithm=MealPyOptimiserBase.CONSTRCUTORS["OriginalBBOA"], 	# Keys in the CONSTURCTORS are algorithm ames
	varType=MealPyOptimiserBase.FLOAT_VAR				# It's a continuous problem. 
)
result		= optimiser.solve()
print(optimiser.lastResult.target.fitness)
###########################
# Configurable properties #
###########################
optimiser.epochs		# int number of epochs
optimiser.minMax		# string [min]imise or [max]imise
optimiser.varType		# MealPy variable type. Typically IntegerVar or FloatVar
optimiser.inequality		# float threshold for classifying the solution as optimal
optimiser.population		# int population size
optimiser.customParams		# Dict of extra problem parameters. E.g obj_weights for multi-objective
optimiser.algorithm		# The MealPy "magic" alogrithm of your choice
````
### Letting the Optimiser select the best MealPy algorithm
````Python
bestAlgorithm	= optimiser.barrage()  # Grab a coffe and let Duckula figure it out.
````

## Supervised learning
Overview video [here](https://www.youtube.com/watch?v=Zr72YpbQ7BA): 
### Using pre-made estimators 
```Python
# Import pre-built XGBoost estimator
from lib.estimators	import XGBoostEstimator
# Load data and create the estimator. Any numeric-only dataset with column headers will do
estimator	= XGBoostEstimator.QuickLoad(PATH_TO_ANY_NUMERIC_DATA_CSV, TARGET_COLUMN_NAME)
# Train with default settings
estimator.train()
# Test the estimator's performance
scores		= estimator.test()	# -> {"rmse": <float>, "mae": <float>, "r2": <float>}
###########################
# Configurable properties #
###########################
## See: https://xgboost.readthedocs.io/en/stable/parameter.html
estimator.booster		# string Boosting algorithm
estimator.maxDepth 		# int Max branches
estimator.learningRate		# float learning rate: solution space step size or something to that effect 
estimator.objective		# string determines regression or classification and score metric. E.g. "reg:sqaurederror"
estimator.sampleType		# string sampling method. xgboost constructor property. E.g uniform
estimator.normaliseType		# string normalisation method. xgboost constructor property
estimator.rateDrop		# float 0 < x < 1 probability that a learner are dropped during an iteration. xgboost constructor property
estimator.skipDrop		# float 0 < x < 1 probability rate drop will be ignored
estimator.nRounds		# int No. of rounds. Basically No. estimators from Random forest or GBDT
estimator.gamma			# float min loss reduction required before further paritioning a leaf 
```
### Custom estimator (sklearn.SVR example)
```Python
### Includes ###
## Native
from sklearn.svm				import SVR
from pandas						import read_csv
## Project
from lib.estimator_base			import EstimatorBase
from lib.hyperopt_hp_tuner_base	import HyperoptHpTunerBase
### Define the estimator  ###
class Estimator(EstimatorBase):
	##
	# data:		DataFrame here. Can be anything with a little class customisation
	# target:	string name of a column in the training data
	##
	def __init__(self, data, target):
		super().__init__(data, target)
		self.C			= 1 	# float regularization parameter. Trade-off between accuracy and model complexity
		self.epsilon	= 0.05	# float tolerance of error before penalising
	##
	# Define the object parameters dictionary passed to the SVR constructor with the **optimiser.allParams dict to parameter thing
	##
	@property
	def params(self):
		return {"C": self.C, "epsilon": self.epsilon}
	##
	# Make an SVR for optimiser.model and train it using optimiser.trainingInputs and optimiser.trainingTargets
	##
	def train(self):
		self.model	= SVR(**self.allParams)
		self.model.fit(self.trainingInputs, self.trainingTargets)
### Create an instance ###
# Load a DataFrame and create an instance
estimator	= XGBoostEstimator.QuickLoad(PATH_TO_ANY_NUMERIC_DATA_CSV, TARGET_COLUMN_NAME)
# It's SVR so let's enable normalisers and scalers
estimator.applyNormaliser	= True
estimator.applyScaler		= True
# Train the model
estimator.train()
```
### Hyperparameter tuning (SVREstimator example)
```Python
### Includes ###
## Project
from lib.hyperopt_hp_tuner_base	import HyperoptHpTunerBase
# Create the estimator object
class Tuner(HyperoptHpTunerBase):
	##
	# We only need to define the evaluation function that updates the model
	# and returns is fitness.
	##
	def evaluate(self):
		self.model.train()
		return self.model.test()["rmse"]
# Create a tuner instance
tuner	= Tuner(estimator)
## Add some tunable parameters
tuner.addUniformIntParameter("C", 1, 60)
tuner.addUniformParameter("epsilon", 0.001, 0.03)
# Tune the SVREstimator
tuner.tune()
### Apply best parameters##
estimator.C		= tuner.best["C"]
estimator.epsilon	= tuner.best["epsilon"]
# Retrain the estimator
estimator.train()
###########################
# Configurable properties #
###########################
tuner.iterations	# int number of stages in the tuning process
tuner.algorithm		# Hyperopt tuning algorithm
tuner.trials		# Hyperopt Trials result tracking object
tuner.optimiser		# Hyperopt fmin or similar
tuner.cvSteps		# int number of cross-validation steps
```
## Credits:
- MealPy for optimisation algorithms: https://github.com/thieu1995/mealpy
- Hyperopt for parameter tuning: https://github.com/hyperopt/hyperopt
