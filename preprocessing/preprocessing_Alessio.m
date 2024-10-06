close all
clear all
clc

addpath('/Users/alessioguarachi/Desktop/stroke')
addpath('/Users/alessioguarachi/Desktop/stroke/functions')

load(['P1_pre_test.mat'])


% Find transitions from 0 to +1
positive_transition_indices = find(trig(1:end-1) == 0 & trig(2:end) == 1) + 1;

% Find transitions from 0 to -1
negative_transition_indices = find(trig(1:end-1) == 0 & trig(2:end) == -1) + 1;

timesteps = 2048;

% form the final dataset of size 80 * 16 size

dataset_pos = zeros(40,2048,16);
for id = 1:40
    dataset_pos(id,:,:) = reshape(y(positive_transition_indices(id):positive_transition_indices(id)+timesteps-1,:),2048,16);
end

dataset_neg = zeros(40,2048,16);
for id = 1:40
    dataset_neg(id,:,:) = reshape(y(negative_transition_indices(id):negative_transition_indices(id)+timesteps-1,:),2048,16);
end

dataset_final = cat(1,dataset_pos,dataset_neg);

% save P1_post_training_epoched.mat dataset_final