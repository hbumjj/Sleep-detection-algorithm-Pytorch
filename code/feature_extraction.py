import os 
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import welch

# rms
def calculate_rms(data):
    return np.sqrt(np.mean(np.square(data)))

# variation
def calculate_variation_vector(signal):
    return np.diff(signal, axis=0)

# fft-1
def get_features(y, T, N, fs, feature_type='psd'):
    if feature_type == 'signal':
        return np.arange(0, len(y)) / fs, y
    elif feature_type == 'fft':
        f_values = fftfreq(N, d=1/fs)[:N//2]
        fft_values = 2.0/N * np.abs(fft(y)[:N//2])
        return f_values, fft_values
    elif feature_type == 'autocorr':
        a_values = np.correlate(y, y, mode='full')[len(y)-1:]
        return np.arange(0, N) * T, a_values
    elif feature_type == 'psd':
        return welch(y, fs)
    else:
        raise ValueError("Invalid feature type")

# fft-2
def feature_extraction(data):
    N, fs, t = 900, 30, 30
    T = t / N
    features = []

    for axis in range(3):
        f_val, fft_values = get_features(data[:, axis], T, N, fs, feature_type='psd')
        features.extend([
            max(fft_values),  # Dominant frequency
            sum(fft_values) / len(fft_values),  # Power
        ])

    return features

# segment processing
def process_file(file_path, save_path):
    raw_down_data = np.load(file_path)
    features = []

    for index in range(0, len(raw_down_data), 900):
        seg_data = raw_down_data[index:index + 900]
        features.append(feature_extraction(seg_data))

    features = np.array(features)
    print(f"Feature shape: {features.shape}")
    np.save(save_path, features)

# main - save
def main():
    input_path = 'PATH/DATA/'
    output_path = 'PATH/FEATURE/'
    
    for file_name in os.listdir(input_path):
        input_file = os.path.join(input_path, file_name)
        output_file = os.path.join(output_path, file_name)
        process_file(input_file, output_file)

if __name__ == "__main__":
    main()