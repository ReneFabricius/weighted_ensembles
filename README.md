# Weighted LDA Ensemble
Weighted LDA Ensemble is a multiclass classification ensembling algorithm based on Linear Discriminant Analysis (LDA) and pairwise coupling.
It is able to combine variable number of multiclass probabilistic classifiers producing probabilistic classification in the same set of classes.
This ensembling method employs LDA model for combining every pair of classes, therefore, it requires training.

## Algorithm description
WeightedLinearEnsemble training and prediction phases can be described by the following diagram.
<img src="https://github.com/ReneFabricius/weighted_ensembles/blob/master/Weighted%20Linear%20Ensemble%20Flowchart.svg" height="800" />  
First, the ensemble constituing probabilistic classifiers need to be obtained. This can be done either by training them, or by using pretrained models.
Then we need to obtain probabilistic classifications from these constituent models for samples in the LDA training dataset.
Then we fit the LDA models using these predictions. For fitting the models we can use method *fit* if we are using softmax outputs of the constituent classifiers, 
or preferably we can use *fit_penultimate* if we have available outputs of the penultimate layer ie. inputs into the softmax function.
These LDA models together with constituent classifiers and pairwise coupling method form the ensemble model.

Prediction is performed by obtaining the probabilistic classifications of the constituing classifiers, computing pairwise probability matrix for each of these classifiers,
combining these matrices by LDA models and then finally by computing multiclass probabilistic classification by selected pairwise coupling method.
Prediction is performed by supplying constituent classifiers outputs into the method *predict_proba*. If we are working with a dataset with
large number of classes, method *predict_proba_topl_fast* may be used. This method considers only *l* (supplied as a parameter) most probable classes from each 
of the constituent classifiers and performs the combination on the union of these classes. Combination computation is also performed in a more optimized manner than in the method *predict_proba*.

## Usage
Weighted LDA Ensemble can be used in custom applications by importing the WeightedLinearEnsemble class from WeightedLinearEnsemble.py.
There is also a script prepared for combining probability classifiers given their outputs on a set for LDA training and their outputs on testing set.
This script is available in the file general_test.py as the function ensemble_general_test.

Examples of usage of the Weighted LDA Ensemble are avilable in the repositories https://github.com/ReneFabricius/cifar_ens_2021 and https://github.com/ReneFabricius/ILSVRC2012_ens.
