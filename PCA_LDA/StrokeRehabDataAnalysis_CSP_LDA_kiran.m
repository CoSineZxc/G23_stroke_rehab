%% CSP + LDA

traininglist = {"P1_pre_training.mat",...
    "P2_pre_training.mat",...
    "P3_pre_training.mat",...
    "P1_post_training.mat",...
    "P2_post_training.mat",...
    "P3_post_training.mat"};

testinglist = {"P1_pre_test.mat",...
    "P2_pre_test.mat",...
    "P3_pre_test.mat",...
    "P1_post_test.mat",...
    "P2_post_test.mat",...
    "P3_post_test.mat"};

actions = {"train","test"};

accuracy = [];

for datasetid = 1:length(traininglist)
    for act = 1:length(actions)

        % load required dataset
        if actions{act} == "train"
            load(traininglist{datasetid})
        else
            load(testinglist{datasetid})
        end
    
        % Initial preprocessing
        timesteps = 2048; % with 256 Hz sampling freq, equates to 8 seconds
        channelno = 16; % 16 channels over somatomotor cortex
        
        % Find transitions from 0 to +1 (total 40 occurrences)
        positive_transition_indices = find(trig(1:end-1) == 0 & trig(2:end) == 1) + 1;
        
        % Find transitions from 0 to -1 (total 40 occurrences)
        negative_transition_indices = find(trig(1:end-1) == 0 & trig(2:end) == -1) + 1;
        
        % separately prepare +1 cases and -1 cases
        dataset_pos = zeros(40,timesteps,channelno);
        for id = 1:40
            dataset_pos(id,:,:) = reshape(y(positive_transition_indices(id):positive_transition_indices(id)+timesteps-1,:),timesteps,channelno);
        end
        dataset_neg = zeros(40,timesteps,channelno);
        for id = 1:40
            dataset_neg(id,:,:) = reshape(y(negative_transition_indices(id):negative_transition_indices(id)+timesteps-1,:),timesteps,channelno);
        end
        
        % form the final dataset of size 80 * 16 size
        dataset_final = cat(1,dataset_pos,dataset_neg);
        
        % label the dataset
        labels = [ones(40,1); 2 * ones(40,1)]; % 1 is +1 (left hand cue), 2 is -1 (right hand cue)
        
        % 'dataset_final' is 80x2048x16 matrix (incorrect dimension: trials x time points x channels).
        % Reshape the matrix to be in the correct format: channels x time points x trials
        eeg_data_correct = permute(dataset_final, [3, 2, 1]);
        
        % Initialize the FieldTrip data structure
        data = [];
        % Set the channel labels (for 16 channels)
        data.label = arrayfun(@(x) ['Ch' num2str(x)], 1:size(eeg_data_correct, 1), 'UniformOutput', false);
        % Set the trial data (each trial is a 16x2048 matrix)
        data.trial = cell(1, size(eeg_data_correct, 3));
        for trial_idx = 1:size(eeg_data_correct, 3)
            data.trial{trial_idx} = eeg_data_correct(:, :, trial_idx);
        end
        % Set the time vector (assuming 256 Hz sampling rate, adjust if different)
        data.time = arrayfun(@(x) (0:(2047))/256, 1:80, 'UniformOutput', false); % 80 trials
        % Set other necessary metadata
        data.fsample = 256; % Sampling rate in Hz
        data.sampleinfo = [1 2048]; % Start and end sample for each trial
        
        % Configure the band-pass filter
        cfg = [];
        cfg.bpfilter = 'yes';       % Enable band-pass filter
        cfg.bpfreq = [8 30];        % Set frequency range (8-30 Hz)
        cfg.channel = 'all';        % Apply to all channels
        filtered_data = ft_preprocessing(cfg, data);  % Filtered EEG data
        
        % Extract time window between time points 512 and 896 for each trial
        time_range = 512:896; % 2 seconds to 3.5 seconds
        
        % Create a new structure for the cropped data
        cropped_data = filtered_data;
        for trial_idx = 1:length(filtered_data.trial)
            cropped_data.trial{trial_idx} = filtered_data.trial{trial_idx}(:, time_range);  % Keep only the selected timepoints
            cropped_data.time{trial_idx} = filtered_data.time{trial_idx}(time_range);       % Adjust the time vector
        end
        
        % Configure CSP
        cfg = [];
        cfg.method = 'csp';
        cfg.csp.classlabels = labels;  % The class labels for each trial (1 or -1)
        csp_filters = ft_componentanalysis(cfg, cropped_data);  % Compute CSP filters
        
        % Initialize a feature matrix
        num_trials = length(cropped_data.trial);
        features = zeros(num_trials, size(csp_filters.unmixing, 1));  % Preallocate feature matrix
        
        for trial_idx = 1:num_trials
            % Apply the CSP filters to the EEG data for each trial
            csp_projected = csp_filters.unmixing * cropped_data.trial{trial_idx};
            
            % Calculate the log-variance of the projected signals as features
            features(trial_idx, :) = log(var(csp_projected, 0, 2));
        end
        
        if actions{act} == "train"
            % Train the LDA classifier using MATLAB%s fitcdiscr function
            lda_model = fitcdiscr(features, labels);
        else
            % Test the model
            % Predict class labels for the training data
            predicted_labels = predict(lda_model, features);
            
            % Calculate accuracy
            modelaccuracy = sum(predicted_labels == labels) / length(labels);
            fprintf('Classification Accuracy: %.2f%%\n', modelaccuracy * 100);
        end
    end
    accuracy = [accuracy; modelaccuracy];
end

disp("Accuracy for all models")
disp(accuracy)