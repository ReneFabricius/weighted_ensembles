# Weighted Linear Ensemble
Weighted Linear Ensemble is a multiclass classification ensembling algorithm based on weighted combination of class pairs pairwise probabilities and pairwise coupling.
It is able to combine variable number of multiclass probabilistic classifiers producing probabilistic classification in the same set of classes.
This ensembling method employs several trainable linear classifier models to combine pairwise probabilities from different ensemble constituents, therefore, it requires training.

## Algorithm description
WeightedLinearEnsemble training and prediction phases can be described by the following diagram.
<img src="https://github.com/ReneFabricius/weighted_ensembles/blob/master/Weighted%20Linear%20Ensemble%20Flowchart.svg" height="800" />  
First, the ensemble constituing probabilistic classifiers need to be obtained. This can be done either by training them, or by using pretrained models.
Then we need to obtain probabilistic classifications from these constituent models for samples in the ensemble training dataset.
Then we fit the linear models using these predictions. For fitting the models we use method *fit* on the outputs of the penultimate layer, ie. inputs into the softmax function, of the constituent classifiers.
These linear models together with constituent classifiers and pairwise coupling method form the ensemble model.

Prediction is performed by obtaining the probabilistic classifications of the constituing classifiers, computing pairwise probability matrix for each of these classifiers,
combining these matrices by linear models and then finally by computing multiclass probabilistic classification by selected pairwise coupling method.
Prediction is performed by supplying constituent classifiers outputs into the method *predict_proba*. If we are working with a dataset with
large number of classes, parameter *l* may be used. This parameter makes the prediction method consider only *l* most probable classes from each 
of the constituent classifiers and performs the combination on the union of these classes.

## Usage
Weighted Linear Ensemble can be used in custom applications by importing the WeightedLinearEnsemble class from WeightedLinearEnsemble.py.
Usage of the algorithm is demonstrated in a colab notebook https://colab.research.google.com/drive/1dRccNaxzeRmPnXDKOzvXLmXs1Ovg2Tu3?usp=sharing

Experiments with the Weighted Linear Ensemble are avilable in the repositories https://github.com/ReneFabricius/cifar_ens_2021 and https://github.com/ReneFabricius/ILSVRC2012_ens.
Demo of the ensemble combining four networks trained on ImageNet1k is hosted at https://huggingface.co/spaces/fabricius/WLEnsemble.
