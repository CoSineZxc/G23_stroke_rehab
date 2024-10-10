close all
clear all
clc

addpath('/Users/alessioguarachi/Desktop/stroke')
addpath('/Users/alessioguarachi/Desktop/stroke/functions')

load(['P1_pre_test.mat'])

X = y';
srate = fs;

[n, m]=size(X); % n=number of channels=16; m=

%% linear detrending
Xdetrended=detrend(X')'; 
X=Xdetrended; %at each step, X is overwritten 

%% Plot of the EEG signals
t=[0:1:m-1]/srate;

start=1;
stop=m;   % full trace

%use the following trick to visualize all signals in the same plot

delta=100;

figure % entire duration
for i = 1:n
    plot(t(start:stop), X(i,start:stop)-delta*(i-1),'color',[0 0 0],'linewidth',1);
    hold on
end
xlim([t(start) t(stop)])
ylim([-delta*n delta])
xlabel('sec')
ylabel('\muV')
title('detrended EEG signals')
set(gca,'ytick',[-delta*(n-1):delta:0])
%set(gca,'ytickLabel',fliplr(ch_names))
set(gca,'fontsize',11)
grid

%% Plot of the triggers over time
figure
plot(trig)

%% Power Spectral Density of the 16 channels

window=srate*10; % window of 10 seconds 
NFFT=window;  % not used zero padding here since in this case it introduces artifacts in the power spectra

start=1;
stop=m; %entire duration

[PSD_unfiltered,f]=pwelch(X(:,start:stop)',window,[],NFFT,srate); 

figure
for i = 1:16
    subplot(4,4,i)
    plot(f(1:end), PSD_unfiltered(1:end,i),'linewidth',1);
    %title(ch_names{i})
    xlim([0 80])
    ylim([0 100])
    xlabel('Hz')
    ylabel('{\muV}^2/Hz')
end
sgtitle('PSD unfitered signals')

%% Band pass filtering 

%LOW-PASS FILTERING
Wp = [30]/(srate/2); 
Ws = [40]/(srate/2); 
Rp = 0.1;
Rs = 40; % attenutation of 40 dB in the stop band
[N,Wp] = ellipord(Wp,Ws,Rp,Rs);
[b,a] = ellip(N,Rp,Rs,Wp);
%%uncomment following rows to visualize the frequency response of the filter
figure
freqz(b,a,srate*10,srate)
title('Frequency response Low Pass Filter')
X_IIR_lp=filtfilt(b,a,X')'; %with filtfilt is a zero-phase filter;
X=X_IIR_lp;

%HIGH-PASS FILTERING
Wp = [8]/(srate/2);    
Ws = [7]/(srate/2); 
Rp = 0.1;
Rs = 40; % attenuation of 40 dB in the stop band
[N,Wp] = ellipord(Wp,Ws,Rp,Rs);
[b,a] = ellip(N,Rp,Rs,Wp,'high');
%%uncomment following rows to visualize the frequency response of the filter
figure
freqz(b,a,srate*10,srate)
title('Frequency response High Pass Filter')
X_IIR_hp=filtfilt(b,a,X')'; %with filtfilt is a zero-phase filter;
X=X_IIR_hp;


%% Power Spectral Density of the filtered signals
window=srate*10; % window of 10 seconds 
NFFT=window;  % not used zero padding here since in this case it introduces artifacts in the power spectra
NOVERLAP = [];

plot_PSDs(X,window, NFFT,NOVERLAP,NFFT, srate);

%% Extraction epochs

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