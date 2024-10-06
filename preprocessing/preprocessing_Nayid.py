# Import necessary libraries
import scipy.io
import numpy as np
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.signal import butter, lfilter, welch
from scipy.linalg import eigh

# File paths for participants P1, P2, and P3
pre_training_path = 'P1_pre_training.mat'
p2_pre_training_path = 'P2_pre_training.mat'
p3_pre_training_path = 'P3_pre_training.mat'

# Load the data for P1, P2, and P3
pre_training_data = scipy.io.loadmat(pre_training_path)
p2_pre_training_data = scipy.io.loadmat(p2_pre_training_path)
p3_pre_training_data = scipy.io.loadmat(p3_pre_training_path)

# Extract EEG data and trigger events
y_pre_training = pre_training_data['y']
trig_pre_training = pre_training_data['trig']

y_p2_pre_training = p2_pre_training_data['y']
trig_p2_pre_training = p2_pre_training_data['trig']

y_p3_pre_training = p3_pre_training_data['y']
trig_p3_pre_training = p3_pre_training_data['trig']

# Preprocessing parameters
lowcut = 8
highcut = 30
sampling_freq = 256
start_timestep = 512  # Start after 2 seconds (2 * 256 Hz)
end_timestep = 896  # End at 3.5 seconds (3.5 * 256 Hz)
num_trials = 80  # Number of trials per session (40 left, 40 right)

# Define bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

# Determine the start of the trials dynamically based on trigger data
def find_first_trial_start(trig_data):
    # Assuming non-zero values in trig_data indicate the start of trials
    trial_starts = np.where(trig_data != 0)[0]
    if len(trial_starts) > 0:
        return trial_starts[0]
    else:
        raise ValueError("No trial start found in trigger data.")

# Find the start index for each dataset
initial_delay_pre_training = find_first_trial_start(trig_pre_training)
initial_delay_p2_pre_training = find_first_trial_start(trig_p2_pre_training)
initial_delay_p3_pre_training = find_first_trial_start(trig_p3_pre_training)

# Remove initial no-activity data before applying the bandpass filter
y_pre_training = y_pre_training[initial_delay_pre_training:]
y_p2_pre_training = y_p2_pre_training[initial_delay_p2_pre_training:]
y_p3_pre_training = y_p3_pre_training[initial_delay_p3_pre_training:]

# Apply bandpass filter to data
filtered_y_pre_training = bandpass_filter(y_pre_training, lowcut, highcut, sampling_freq)
filtered_y_p2_pre_training = bandpass_filter(y_p2_pre_training, lowcut, highcut, sampling_freq)
filtered_y_p3_pre_training = bandpass_filter(y_p3_pre_training, lowcut, highcut, sampling_freq)

# Define function to extract relevant motor imagery segment (2 to 3.5 seconds)
def extract_motor_imagery_segment(filtered_data, trig_data, num_trials, start_timestep, end_timestep, initial_delay):
    trials = []
    labels = []
    data_length = filtered_data.shape[0]
    segment_length = end_timestep - start_timestep

    for i in range(num_trials):
        trial_start_idx = i * (end_timestep + start_timestep)  # Adjust trial start index accordingly
        segment_start_idx = trial_start_idx + start_timestep
        segment_end_idx = segment_start_idx + segment_length

        # Check if the end index exceeds the data length
        if segment_end_idx > data_length:
            break

        trial_data = filtered_data[segment_start_idx:segment_end_idx, :]
        trials.append(trial_data)

        label = trig_data[initial_delay + segment_start_idx, 0]
        labels.append(label)

    return np.array(trials), np.array(labels)

# Extract motor imagery segments for P1, P2, and P3
trials_pre_training_segment, labels_pre_training = extract_motor_imagery_segment(
    filtered_y_pre_training, trig_pre_training, num_trials, start_timestep, end_timestep, initial_delay_pre_training
)

trials_p2_segment, labels_p2_pre_training = extract_motor_imagery_segment(
    filtered_y_p2_pre_training, trig_p2_pre_training, num_trials, start_timestep, end_timestep, initial_delay_p2_pre_training
)

trials_p3_segment, labels_p3_pre_training = extract_motor_imagery_segment(
    filtered_y_p3_pre_training, trig_p3_pre_training, num_trials, start_timestep, end_timestep, initial_delay_p3_pre_training
)
