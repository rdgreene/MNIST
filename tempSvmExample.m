%% INIT
clear; clc; close all

%% LOAD & TABULATE DATA

% load MNIST data set using Stanford's functions
trainingData = loadMNISTImages('train-images.idx3-ubyte');
trainingLabels = loadMNISTLabels('train-labels.idx1-ubyte');

%% SHUFFLE AND PROCESS TRAINING DATASET

% randomly shuffle data (rng seeding used for reproducibility)
rng(17); shuffle = randperm(size(trainingData, 2));
trainingData = trainingData(:, shuffle);
trainingLabels = trainingLabels(shuffle);

%% Run K-Fold Cross Validation

[trainError valError] = svmKFoldValidate( 3, trainingData, trainingLabels);






   