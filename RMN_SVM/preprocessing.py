import mne
import os
from scipy import signal
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import pandas as pd

def common_average_reference(EEG_data):
    # Calculate the average across all channels (row-wise average)
    mean_across_channels = np.mean(EEG_data, axis=0)
    
    # Subtract the average from each channel
    re_referenced_data = EEG_data - mean_across_channels
    
    return re_referenced_data

def extract_keywords(filename):
    # Define the regex pattern
    pattern = r"(P\d+)_(pre|post)_(test|training)\.mat"
    
    # Search for the pattern in the filename
    match = re.match(pattern, filename)
    
    if match:
        # Extract the components (P1, pre/post, test/training)
        PID= match.group(1)  # P1, P2, P3, etc.
        stage = match.group(2)        # pre or post
        purpose = match.group(3)    # test or training
        
        return PID, stage, purpose
    else:
        return None  # If pattern doesn't match

def cct_PSD_meanstd(EEG,fs):
    print(EEG.shape)
    PSD=[]
    if len(EEG.shape)==2:
        for chl_idx,chl in enumerate(EEG):
            f, Pxx_spec = signal.welch(chl, fs, 'hann', 256*2, scaling='density')
            PSD.append(Pxx_spec)
    elif len(EEG.shape)==3:
        for trial_idx, trial in enumerate(EEG):
            for chl_idx,chl in enumerate(trial):
                f, Pxx_spec = signal.welch(chl, fs, 'hann', 256*2, scaling='density')
                PSD.append(Pxx_spec)
    psd_array=np.array(PSD)
    psd_array_uv2 = psd_array #* 1e12  # Convert to µV^2/Hz
    psd_array_db = 10 * np.log10(psd_array_uv2)  # Convert to dB
    mean_psd = np.mean(psd_array_db, axis=0)
    std_psd = np.std(psd_array_db, axis=0)
    return f,mean_psd,std_psd

def plot_PSD(freqs,mean_psd,std_psd,title):
    # plt.semilogy(freqs, mean_psd)
    plt.plot(freqs, mean_psd)
    plt.fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd, alpha=0.3, label='±1 Std Dev')
    plt.xlim([0,128])
    # plt.ylim([1e-19,1e-8])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (µV^2/Hz) [dB]')
    plt.title(title)
    plt.show()
    
def multi_plot(PSD_info_list,noise_detect_mtx,x_num,y_num,Suptitle,lowband,highband):
    plt.figure(figsize=(10, 6))
    for PSD_idx,[freqs,mean_psd,std_psd,title] in enumerate(PSD_info_list):
        plt.subplot(x_num,y_num,PSD_idx+1)
        plt.plot(freqs, mean_psd)
        plt.fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd, alpha=0.3, label='±1 Std Dev')
        plt.xlim([0,128])
        plt.ylim([-30,30])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (µV^2/Hz) [dB]')
        plt.title(title)
    plt.subplot(x_num,y_num,len(PSD_info_list)+1)
    plt.imshow(noise_detect_mtx, cmap='viridis', interpolation='nearest')
    plt.title('Noise removal')
    plt.suptitle(Suptitle+f"_{lowband}_{highband}")
    plt.tight_layout()
    # plt.show()
    if os.path.exists('./Pre_img/')==False:
        os.makedirs('./Pre_img/')
    plt.savefig(f"./Pre_img/{Suptitle}_{lowband}_{highband}_preprocessing.png")
    plt.close()

def notch_filter(data, fs, f0=50.0, Q=30.0):
    # Design the notch filter
    b, a = signal.iirnotch(f0, Q, fs)

    # Apply the filter to the data
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    
    return filtered_data

def butter_bandpass_filter(data, lowband, highband, Fre, order=4):
    b, a = signal.butter(order, [lowband, highband],fs=Fre, btype='band')
    filtered_data = np.zeros_like(data)
    for j in range(data.shape[0]):  # Loop over channels
        filtered_data[j, :] = signal.filtfilt(b, a, data[j, :])
    return filtered_data

def epoch_data(EEG, eventlist, t_start, t_end, fre):
    n_channels = EEG.shape[0]
    
    n_epochs=len(eventlist)
    
    n_samples_start=int(t_start*fre)
    n_samples_end=int(t_end*fre)
    n_samples_epoch = int(n_samples_end - n_samples_start)
    
    epochs = np.zeros((n_epochs, n_channels, n_samples_epoch))
    for event_i, event in enumerate(eventlist):
        # Define the window of data to extract for each trigger
        epoch_start = event + n_samples_start
        epoch_end = event + n_samples_end
        
        # Check if the epoch falls within valid bounds
        if epoch_start >= 0 and epoch_end <= EEG.shape[1]:
            # Extract the epoch data
            epochs[event_i, :, :] = EEG[:, epoch_start:epoch_end]
        else:
            print(f"Epoch {event_i} is out of bounds and will be skipped.")
    return epochs

def Create_eventlist_labellist(trigger_list,fs):
    start_label=False
    eventlist=[]
    labellist=[]
    for tri_idx, trigger in enumerate(trigger_list):
        if trigger==0 and start_label==False:
            continue
        elif trigger!=0 and start_label==False:
            eventlist.append(tri_idx+2*fs)
            labellist.append(trigger)
            start_label=True
        elif trigger!=0 and start_label==True:
            continue
        elif trigger==0 and start_label==True:
            start_label=False
        
    return eventlist,labellist

def detect_trial_type_bi(array,voltage_threshold):
    abs_array=np.abs(array)
    above_threshold=abs_array>voltage_threshold
    count_dict=Counter(above_threshold)
    if count_dict[True]==0:     # nice signal
        return True
    else:   # polluted signal
        return False

def Hard_threshold_removal(EEG,hard_threshold,removal_perc,chl_name,fs,labellist):
    detect_mtx=np.zeros((EEG.shape[1], EEG.shape[0]))
    clean_epochs = []
    num_chl=EEG.shape[1]
    epochs_to_remove = []
    interp_item_num=0
    clean_label_list=[]
    for epoch_idx, trial in enumerate(EEG):
        bad_channels=[]
        for chl_idx, channel_data in enumerate(trial):
            data_type=detect_trial_type_bi(channel_data,hard_threshold)
            if data_type==True: # nice channel data
                continue
            else:               # polluted data
                bad_channels.append(chl_idx)
        Count_bad=len(bad_channels)
        # print(Count_bad)
        if Count_bad>=removal_perc*num_chl:
            epochs_to_remove.append(epoch_idx)
            detect_mtx[:,epoch_idx]=2
        elif 0<Count_bad<removal_perc*num_chl:
            interp_item_num+=len(bad_channels)
            detect_mtx[bad_channels,epoch_idx]=1
            bad_chl=[chl_name[i] for i in bad_channels]
            # print(bad_chl)
            info = mne.create_info(ch_names=chl_name, sfreq=fs, ch_types='eeg')
            raw = mne.io.RawArray(trial, info)
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage)
            raw.info['bads'] = bad_chl
            # print(type(temp_epoch.get_data()))
            trial_cleaned = raw.interpolate_bads()#reset_bads=True)#, mode='accurate')
            clean_epochs.append(trial_cleaned.get_data())
            clean_label_list.append(labellist[epoch_idx])
        else:
            clean_epochs.append(EEG[epoch_idx])
            clean_label_list.append(labellist[epoch_idx])
    
    cleaned_epochs_np = np.array(clean_epochs)
    remove_num=len(epochs_to_remove)
    perc_interp=interp_item_num/(cleaned_epochs_np.shape[0]*cleaned_epochs_np.shape[1])
    return cleaned_epochs_np,detect_mtx,remove_num,perc_interp,clean_label_list
    # return cleaned_data,detect_mtx,remove_num,perc_interp

if __name__=="__main__":
    mne.set_log_level('ERROR')
    Raw_EEG_folder='./'
    Hard_threshold=100 #100µV
    threshold_removal_perc = 1/3
    chl_name=['FC3','FCz','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1',
              'CPz','CP2','CP4','Pz']
    part_info=[]
    for raw_mat_file in os.listdir(Raw_EEG_folder):
        if '.mat' not in raw_mat_file:
            continue
        else:
            raw_data_dict = scipy.io.loadmat(raw_mat_file)
        # pid_selected='P1'
        ###-------- load data and get PID --------
        PID,stage,purpose=extract_keywords(raw_mat_file)
        # if PID!=pid_selected:
        #     continue
        raw_EEG_data=raw_data_dict['y'].T
        samplingrate=raw_data_dict['fs'][0,0]
        print(f"{PID} {stage} {purpose} loading data...")
        for [lowband,highband] in [[8,13],[13,30],[8,30],[30,40]]:#[[8,30]]:#
            psd_info_list=[]
            print(f"Fre band: {lowband} - {highband}")
            ###-------- rereference --------
            ### Common Average Referencing
            print("Re-referencing...")
            avg_ref_EEG=common_average_reference(raw_EEG_data)
            freqs_psd,mean_psd,std_psd=cct_PSD_meanstd(avg_ref_EEG,samplingrate)
            psd_info_list.append([freqs_psd,mean_psd,std_psd,'raw'])
            # plot_PSD(freqs_psd,mean_psd,std_psd,'raw')
            ###-------- remove line noise --------
            ### only focus on 50 Hz cause our research only focus on 1-48 Hz
            print("Removing line noise...")
            line_freq = 50  
            notch_EEG = notch_filter(avg_ref_EEG,fs=samplingrate,f0=line_freq)
            freqs_psd,mean_psd,std_psd=cct_PSD_meanstd(notch_EEG,samplingrate)
            psd_info_list.append([freqs_psd,mean_psd,std_psd,'notch'])
            # plot_PSD(freqs_psd,mean_psd,std_psd,'notch')
            ##-------- band pass filter --------
            ### butterworth filter 8th order zero-phase
            print("Bandpass filtering...")
            band_pass_EEG=butter_bandpass_filter(notch_EEG,lowband,highband,samplingrate,order=4)
            freqs_psd,mean_psd,std_psd=cct_PSD_meanstd(band_pass_EEG,samplingrate)
            psd_info_list.append([freqs_psd,mean_psd,std_psd,'bandpass'])
            # plot_PSD(freqs_psd,mean_psd,std_psd,'bandpass')
            ###-------- epoch data --------
            print("Epoching data...")
            eventlist,labellist=Create_eventlist_labellist(raw_data_dict['trig'],samplingrate)
            epoch_EEG=epoch_data(band_pass_EEG,eventlist,0,3,samplingrate)
            freqs_psd,mean_psd,std_psd=cct_PSD_meanstd(epoch_EEG,samplingrate)
            psd_info_list.append([freqs_psd,mean_psd,std_psd,'epoch'])
            # plot_PSD(freqs_psd,mean_psd,std_psd,'epoch')
            ###-------- Hard threshold detection --------
            ### check data trial by trial, mark channel of which absolute amplitude over 100µV 
            print("Noise removing...")
            threshold_voltage = Hard_threshold  
            clean_epoch_EEG,detect_mtx,remove_num,interp_perc,clean_labellist=Hard_threshold_removal(epoch_EEG,
                                                                                     threshold_voltage,
                                                                                     threshold_removal_perc,
                                                                                     chl_name,samplingrate,
                                                                                     labellist)
            freqs_psd,mean_psd,std_psd=cct_PSD_meanstd(clean_epoch_EEG,samplingrate)
            psd_info_list.append([freqs_psd,mean_psd,std_psd,'noise removal'])
            multi_plot(psd_info_list,detect_mtx,2,3,f"{PID}-{stage}-{purpose}",lowband,highband)
            print('the number of removal: '+str(remove_num))
            print('Interpolation percentage: '+str(interp_perc))
            part_info.append([f"{PID}-{stage}-{purpose}",remove_num,interp_perc])
            ###-------- downsampling --------
            ### downsample data from 1000Hz to 100Hz
            # print("downsampling...")
            # down_EEG_np=downsampleEEG(clean_epoch_EEG,1000,100,0.5)
            if len(clean_labellist)!=clean_epoch_EEG.shape[0]:
                print('ERROR')
            clean_EEG_dict={
                'EEGdata':clean_epoch_EEG,
                'labellist':clean_labellist
                }
            Clean_EEG_dir="./filtered_data/"
            if os.path.exists(Clean_EEG_dir)==False:
                os.makedirs(Clean_EEG_dir)
            filename=f"filtered_EEG_{PID}_{stage}_{purpose}_{lowband}_{highband}.mat"
            with open(Clean_EEG_dir+filename, 'wb') as f:
                scipy.io.savemat(f, clean_EEG_dict)
                
            # clean_epoch_EEG.save("../data/preprocessing_v3/filter_data/"+filename,overwrite=True)
            # epoch_EEG.plot(title='before interpolation')
            # clean_epoch_EEG.plot(title='after interpolation')
        #     break
        # break
    part_info_df=pd.DataFrame(part_info,columns=['PID','removal number','interpolation percentage'])
    part_info_df.to_csv('./preprocessing_result.csv')
    # mne.set_log_level('INFO')