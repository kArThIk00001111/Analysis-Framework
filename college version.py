from flask import Flask, request, render_template, jsonify
from scipy.signal import butter, filtfilt
import os
from werkzeug.utils import secure_filename
from nptdms import TdmsFile
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, cheby1, filtfilt, savgol_filter
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from sklearn.cluster import KMeans
from statistics import mode
from bs4 import BeautifulSoup
from py_mini_racer import py_mini_racer
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES

import json

app = Flask(__name__)

UPLOAD_FOLDER = 'u'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
sel_cycle = ''


def read_tdms_file(file_path):
    try:
        with TdmsFile.read(file_path) as tdms_file:
            data = {}
            for group in tdms_file.groups():
                if group.name == 'GT Signals':
                    for channel in group.channels():
                        if channel.name == 'Power':
                            data[channel.name] = channel[:]
            if 'Power' not in data:
                raise ValueError("Power channel not found in TDMS file")
        return data
    except Exception as e:
        print(f"Error reading TDMS file '{file_path}': {e}")
        return None


def calculate_cutoff_frequency():
    sampling_freq1 = 10
    nyquist_freq = 0.5 * sampling_freq1
    cutoff_freq = min(10, nyquist_freq - 1e-6)
    return cutoff_freq


def apply_butterworth_filter_order_3(original_signal):
    order = 3
    cut_off = calculate_cutoff_frequency()
    fs = 10
    b, a = butter(order, cut_off / (0.5 * fs), btype='low')
    filtered_signal = filtfilt(b, a, original_signal)
    return filtered_signal


def apply_butterworth_filter_order_4(original_signal):
    order = 4
    fs = 10
    cut_off = calculate_cutoff_frequency()
    b, a = butter(order, cut_off / (0.5 * fs), btype='low')
    filtered_signal = filtfilt(b, a, original_signal)
    return filtered_signal


def apply_butterworth_filter_order_5(original_signal):
    order = 5
    cut_off = calculate_cutoff_frequency()
    fs = 10
    b, a = butter(order, cut_off / (0.5 * fs), btype='low')
    filtered_signal = filtfilt(b, a, original_signal)
    return filtered_signal


def apply_lms_filter_order_3(original_signal):
    mu = 0.1
    order = 3

    filtered_signal = np.zeros(len(original_signal))  # Initialize filtered_signal
    error_signal = np.zeros(len(original_signal))  # Initialize error_signal
    w = np.zeros(order)

    for i in range(order, len(original_signal)):
        x = original_signal[i - order:i]
        y_pred = np.dot(w, x)
        error = original_signal[i] - y_pred

        # Gradient clipping to prevent overflow
        gradient = 2 * mu * error * x
        max_gradient = 1e6  # Adjust as needed
        if np.linalg.norm(gradient) > max_gradient:
            gradient *= max_gradient / np.linalg.norm(gradient)

        w += gradient
        filtered_signal[i] = y_pred
        error_signal[i] = error

    return filtered_signal


def apply_lms_filter_order_4(original_signal):
    mu = 0.1
    order = 4

    filtered_signal = np.zeros(len(original_signal))  # Initialize filtered_signal
    error_signal = np.zeros(len(original_signal))  # Initialize error_signal
    w = np.zeros(order)

    for i in range(order, len(original_signal)):
        x = original_signal[i - order:i]
        y_pred = np.dot(w, x)
        error = original_signal[i] - y_pred

        # Gradient clipping to prevent overflow
        gradient = 2 * mu * error * x
        max_gradient = 1e6  # Adjust as needed
        if np.linalg.norm(gradient) > max_gradient:
            gradient *= max_gradient / np.linalg.norm(gradient)

        w += gradient
        filtered_signal[i] = y_pred
        error_signal[i] = error

    return filtered_signal


def apply_chebyshev_type1_filter_order_3(original_signal):
    order = 3
    rp = 0.1
    cutoff_freq = calculate_cutoff_frequency()
    fs = 10
    b, a = cheby1(order, rp, cutoff_freq / (0.5 * fs), btype='low')
    filtered_signal = filtfilt(b, a, original_signal)
    return filtered_signal


def apply_chebyshev_type1_filter_order_4(original_signal):
    order = 4
    rp = 0.1
    cutoff_freq = calculate_cutoff_frequency()
    fs = 10
    b, a = cheby1(order, rp, cutoff_freq / (0.5 * fs), btype='low')
    filtered_signal = filtfilt(b, a, original_signal)
    return filtered_signal


def apply_kalman_filter(original_signal):
    process_noise = 0.1
    measurement_noise = 0.1

    # Kalman filter implementation
    n = len(original_signal)
    x_hat = np.zeros(n)  # Estimated state
    P = np.zeros(n)  # Estimated error covariance
    x_hat_minus = np.zeros(n)  # Predicted state
    P_minus = np.zeros(n)  # Predicted error covariance
    K = np.zeros(n)  # Kalman gain
    Q = process_noise  # Process noise covariance
    R = measurement_noise  # Measurement noise covariance

    # Initial conditions
    x_hat[0] = original_signal[0]
    P[0] = 1.0

    for k in range(1, n):
        # Time update (Prediction)
        x_hat_minus[k] = x_hat[k - 1]
        P_minus[k] = P[k - 1] + Q

        # Measurement update (Correction)
        K[k] = P_minus[k] / (P_minus[k] + R)
        x_hat[k] = x_hat_minus[k] + K[k] * (original_signal[k] - x_hat_minus[k])
        P[k] = (1 - K[k]) * P_minus[k]
    filtered_signal = x_hat

    return filtered_signal


def apply_savitzky_golay_filter(original_signal):
    window_length = 11
    polyorder = 3

    filtered_signal = savgol_filter(original_signal, window_length, polyorder)
    return filtered_signal


def apply_gaussian_filter(original_signal):
    sigma = 2
    filtered_signal = gaussian_filter1d(original_signal, sigma)
    return filtered_signal


def calculate_correlation_coefficient(filtered_signal, original_signal):
    correlation_coefficient = np.corrcoef(original_signal, filtered_signal)[0, 1]
    return correlation_coefficient


def calculate_mean_squared_error(filtered_signal, original_signal):
    mean_squared_error = np.mean((original_signal - filtered_signal) ** 2)
    return mean_squared_error


def best_filter(original_signal):
    filtered_signal_butterworth_order_3 = apply_butterworth_filter_order_3(original_signal)
    corr_coefficient_butterworth_order_3 = calculate_correlation_coefficient(original_signal,
                                                                             filtered_signal_butterworth_order_3)
    mse_butterworth_order_3 = calculate_mean_squared_error(original_signal, filtered_signal_butterworth_order_3)

    filtered_signal_butterworth_order_4 = apply_butterworth_filter_order_4(original_signal)
    corr_coefficient_butterworth_order_4 = calculate_correlation_coefficient(original_signal,
                                                                             filtered_signal_butterworth_order_4)
    mse_butterworth_order_4 = calculate_mean_squared_error(original_signal, filtered_signal_butterworth_order_4)

    filtered_signal_butterworth_order_5 = apply_butterworth_filter_order_5(original_signal)
    corr_coefficient_butterworth_order_5 = calculate_correlation_coefficient(original_signal,
                                                                             filtered_signal_butterworth_order_5)
    mse_butterworth_order_5 = calculate_mean_squared_error(original_signal, filtered_signal_butterworth_order_5)

    filtered_signal_savitzky_golay = apply_savitzky_golay_filter(original_signal)
    corr_coefficient_savitzky_golay = calculate_correlation_coefficient(original_signal, filtered_signal_savitzky_golay)
    mse_savitzky_golay = calculate_mean_squared_error(original_signal, filtered_signal_savitzky_golay)

    filtered_signal_chebyshev_order_3 = apply_chebyshev_type1_filter_order_3(original_signal)
    corr_coefficient_chebyshev_order_3 = calculate_correlation_coefficient(original_signal,
                                                                           filtered_signal_chebyshev_order_3)
    mse_chebyshev_order_3 = calculate_mean_squared_error(original_signal, filtered_signal_chebyshev_order_3)

    filtered_signal_chebyshev_order_4 = apply_chebyshev_type1_filter_order_4(original_signal)
    corr_coefficient_chebyshev_order_4 = calculate_correlation_coefficient(original_signal,
                                                                           filtered_signal_chebyshev_order_4)
    mse_chebyshev_order_4 = calculate_mean_squared_error(original_signal, filtered_signal_chebyshev_order_4)

    filtered_signal_gaussian = apply_gaussian_filter(original_signal)
    corr_coefficient_gaussian = calculate_correlation_coefficient(original_signal, filtered_signal_gaussian)
    mse_gaussian = calculate_mean_squared_error(original_signal, filtered_signal_gaussian)

    filtered_signal_kalman = apply_kalman_filter(original_signal)
    corr_coefficient_kalman = calculate_correlation_coefficient(original_signal, filtered_signal_kalman)
    mse_kalman = calculate_mean_squared_error(original_signal, filtered_signal_kalman)

    filtered_signal_lms_order_3 = apply_lms_filter_order_3(original_signal)
    corr_coefficient_lms_order_3 = calculate_correlation_coefficient(original_signal, filtered_signal_lms_order_3)
    mse_lms_order_3 = calculate_mean_squared_error(original_signal, filtered_signal_lms_order_3)

    filtered_signal_lms_order_4 = apply_lms_filter_order_4(original_signal)
    corr_coefficient_lms_order_4 = calculate_correlation_coefficient(original_signal, filtered_signal_lms_order_4)
    mse_lms_order_4 = calculate_mean_squared_error(original_signal, filtered_signal_lms_order_4)

    mse_values = [mse_butterworth_order_3, mse_butterworth_order_4, mse_butterworth_order_5, mse_savitzky_golay,
                  mse_gaussian, mse_kalman, mse_lms_order_3, mse_lms_order_4, mse_chebyshev_order_3,
                  mse_chebyshev_order_4]

    corr_coefficients = [corr_coefficient_butterworth_order_3, corr_coefficient_butterworth_order_4,
                         corr_coefficient_butterworth_order_5, corr_coefficient_savitzky_golay,
                         corr_coefficient_gaussian, corr_coefficient_kalman, corr_coefficient_lms_order_3,
                         corr_coefficient_lms_order_4, corr_coefficient_chebyshev_order_3,
                         corr_coefficient_chebyshev_order_4]

    # Determine the best filter based on max correlation coefficient and min MSE
    combined_scores = [corr_coefficient + 1 / (mse + 1) for corr_coefficient, mse in zip(corr_coefficients, mse_values)]
    best_filter_index = np.argmax(combined_scores)

    best_filters = ['Butterworth Order 3', 'Butterworth Order 4', 'Butterworth Order 5', 'Savitzky-Golay',
                    'Gaussian', 'Kalman', 'LMS Order 3', 'LMS Order 4', 'Chebyshev Type I Order 3',
                    'Chebyshev Type I Order 4'][best_filter_index + 1]

    # Apply the best filter
    if best_filters == 'Butterworth Order 3':
        return filtered_signal_butterworth_order_3
    elif best_filters == 'Butterworth Order 4':
        return filtered_signal_butterworth_order_4
    elif best_filters == 'Butterworth Order 5':
        return filtered_signal_butterworth_order_5
    elif best_filters == 'Savitzky-Golay':
        return filtered_signal_savitzky_golay
    elif best_filters == 'Gaussian':
        return filtered_signal_gaussian
    elif best_filters == 'Kalman':
        return filtered_signal_kalman
    elif best_filters == 'LMS Order 3':
        return filtered_signal_lms_order_3
    elif best_filters == 'LMS Order 4':
        return filtered_signal_lms_order_4
    elif best_filters == 'Chebyshev Type I Order 3':
        return filtered_signal_chebyshev_order_3
    elif best_filters == 'Chebyshev Type I Order 4':
        return filtered_signal_chebyshev_order_4
    else:
        # Default to Butterworth Order 3 if no specific filter matches
        return 'no'


def time_generation(original_signal):
    start_time_str = '13-08-2020 10:35:06.095 AM'
    sample_rate = 10  # Hz
    num_samples = len(original_signal)
    start_time = datetime.strptime(start_time_str, '%d-%m-%Y %I:%M:%S.%f %p')
    current_time = start_time
    time_samples = []

    for i in range(num_samples):
        time_samples.append((current_time - start_time).total_seconds())
        current_time += timedelta(seconds=1 / sample_rate)

    time_data = pd.DataFrame({'Time_Seconds': time_samples})
    return time_data


def calculate_cycles(original_signal, ideal):
    cycles = 0  # Initialize the cycle count to 0
    in_cycle = False  # Initialize a variable to track whether we're inside a cycle
    above_threshold_count = 0  # Initialize a count for consecutive values above the threshold
    below_threshold_count = 0  # Initialize a count for consecutive values below the threshold
    cycle_avg_powers = []  # List to store average powers of each cycle

    for index, value in enumerate(original_signal):  # Iterate over each value in the original signal
        if value > ideal:  # Check if the value is greater than the ideal
            above_threshold_count += 1
            below_threshold_count = 0
        else:  # If the value is less than or equal to the ideal
            above_threshold_count = 0
            below_threshold_count += 1

        if above_threshold_count >= 10 and value - ideal >= 0.9:  # Check if the power rises above the ideal continuously for 10 times and difference is at least 0.5
            if not in_cycle:  # If we're not already in a cycle
                in_cycle = True  # Set in_cycle to True to indicate that we're inside a cycle
                cycle_total_power = 0  # Reset total power in cycle
                cycle_length = 0  # Reset cycle length

        if below_threshold_count >= 1 and in_cycle:  # Check if the power decreases or equals the ideal value consistently for 10 times and we're inside a cycle
            cycle_avg_power = cycle_total_power / cycle_length  # Calculate average power of the cycle
            cycle_avg_powers.append(
                round(cycle_avg_power))  # Add average power of the cycle to the list of cycle average powers
            in_cycle = False  # Set in_cycle to False to indicate that we're no longer in a cycle
            below_threshold_count = 0  # Reset the count of consecutive values below the threshold

        if in_cycle:  # Update cycle information if we're in a cycle
            cycle_length += 1
            cycle_total_power += value

    # Check if the last cycle ended above the threshold but the signal didn't drop below the ideal value
    if in_cycle:
        cycle_avg_power = cycle_total_power / cycle_length  # Calculate average power of the last cycle
        cycle_avg_powers.append(round(cycle_avg_power))

    if cycle_avg_powers:  # Check if there are any cycles
        avg_cycle_power = mode(cycle_avg_powers)  # Find the mode of the cycle average powers

        # Filter out cycles with significant changes in average power
        filtered_cycle_avg_powers = [avg_power for avg_power in cycle_avg_powers if
                                     abs(avg_power - avg_cycle_power) <= 1]
        cycles = len(filtered_cycle_avg_powers)  # Update cycle count
    return cycles


def calculate_cycle_times(original_signal, time_df, ideal):
    cycles = []
    cycle_avg_powers = []
    threshold_count = 0
    in_cycle = False
    above_threshold_count = 0
    below_threshold_count = 0
    start_time = None

    for index, value in enumerate(original_signal):
        if value > ideal:
            above_threshold_count += 1
            below_threshold_count = 0
        else:
            above_threshold_count = 0
            below_threshold_count += 1

        if above_threshold_count >= 10 and value - ideal >= 0.9:
            if not in_cycle:
                in_cycle = True
                start_time = time_df['Time_Seconds'].iloc[index]
                cycle_total_power = 0  # Initialize total power for the cycle
                cycle_length = 0  # Initialize cycle length

        if below_threshold_count >= 1 and in_cycle:
            end_time = time_df['Time_Seconds'].iloc[index]
            cycle_avg_power = cycle_total_power / cycle_length  # Calculate average power of the cycle
            cycle_avg_powers.append(cycle_avg_power)  # Store average power of the cycle
            cycles.append((start_time, end_time))  # Add the cycle to the list
            in_cycle = False
            below_threshold_count = 0

        if in_cycle:
            cycle_length += 1
            cycle_total_power += value

    if in_cycle:
        end_time = time_df['Time_Seconds'].iloc[-1]
        cycle_avg_power = cycle_total_power / cycle_length
        cycle_avg_powers.append(cycle_avg_power)
        cycles.append((start_time, end_time))

    if cycle_avg_powers:
        avg_cycle_power = mode(cycle_avg_powers)  # Calculate the mode of cycle average powers

        # Filter out cycles with significant changes in average power
        filtered_cycles = [cycle for cycle, avg_power in zip(cycles, cycle_avg_powers) if
                           abs(avg_power - avg_cycle_power) <= 1]
        return filtered_cycles

    return cycles


def get_min_value_matching_mode(filtered_signal):
    # Helper function to format each value to two decimal points
    def format_to_two_decimals(value):
        return round(value, 1)

    # Format each value in the filtered signal to two decimal points
    formatted_signal = [format_to_two_decimals(value) for value in filtered_signal]

    # Calculate the mode of the formatted signal
    mode_counts = Counter(formatted_signal)
    max_count = max(mode_counts.values())
    modes = [value for value, count in mode_counts.items() if count == max_count]

    # Return the minimum value among the modes
    min_matching_value = min(modes)

    return min_matching_value


def find_cycle_powers(original_signal, ideal):
    cycles = []  # Initialize a list to store cycle powers
    current_cycle = []  # Initialize a list to store powers of the current cycle

    in_cycle = False  # Initialize a variable to track whether we're inside a cycle
    above_threshold_count = 0  # Initialize a count for consecutive values above the threshold
    below_threshold_count = 0  # Initialize a count for consecutive values below the threshold

    for index, value in enumerate(original_signal):  # Iterate over each value in the original signal
        if value > ideal:  # Check if the value is greater than the ideal
            above_threshold_count += 1
            below_threshold_count = 0
        else:  # If the value is less than or equal to the ideal
            above_threshold_count = 0
            below_threshold_count += 1

        if above_threshold_count >= 10 and value - ideal >= 0.9:  # Check if the power rises above the ideal continuously for 30 times
            if not in_cycle:  # If we're not already in a cycle
                in_cycle = True  # Set in_cycle to True to indicate that we're inside a cycle
            above_threshold_count = 0  # Reset the count of consecutive values above the threshold

        if below_threshold_count >= 1 and in_cycle:  # Check if the power decreases or equals the ideal value consistently for 30 times and we're inside a cycle
            cycles.append(current_cycle)  # Add the powers of the completed cycle to the cycles list
            current_cycle = []  # Reset the list for the next cycle
            in_cycle = False  # Set in_cycle to False to indicate that we're no longer in a cycle
            below_threshold_count = 0  # Reset the count of consecutive values below the threshold

        if in_cycle:  # If we're inside a cycle
            current_cycle.append(value)  # Add the current power to the list of powers for the current cycle

    # Check if the last cycle ended above the threshold but the signal didn't drop below the ideal value
    if current_cycle:
        cycles.append(current_cycle)

    cycle_powers = []  # Initialize a list to store maximum and average powers of each cycle
    for cycle in cycles:
        max_power = max(cycle)  # Find the maximum power in the cycle
        avg_power = sum(cycle) / len(cycle)  # Calculate the average power in the cycle
        cycle_powers.append((max_power, avg_power))  # Add the maximum and average powers to the list

    return cycle_powers


def process_tdms_signal(original_signal, ideal):
    final_df_list = []  # List to store all merged DataFrames
    num_clusters = 4
    sample_rate = 10

    # Define a dictionary to map cluster labels to part statuses

    # Concatenate all the DataFrames into a single DataFrame
    in_cycle = False
    above_threshold_count = 0
    below_threshold_count = 0
    cycle_data = []
    final_df_list = []
    status_time_list = []

    # Iterate over the signal to detect cycles
    for index, value in enumerate(original_signal):
        if value > ideal:
            above_threshold_count += 1
            below_threshold_count = 0
        else:
            above_threshold_count = 0
            below_threshold_count += 1

        if above_threshold_count >= 5 and value - ideal >= 0.9:
            if not in_cycle:
                in_cycle = True
                current_cycle = []

        if in_cycle:
            current_cycle.append((index, value))

        if below_threshold_count >= 1 and in_cycle:
            in_cycle = False
            cycle_data.append(current_cycle)
            current_cycle = []

    if in_cycle:
        cycle_data.append(current_cycle)

    # Process each cycle
    for cycle_index, cycle in enumerate(cycle_data):
        total_power = sum(value for _, value in cycle)
        cycle_length = len(cycle)
        avg_power = total_power / cycle_length

        if avg_power > 0:  # Consider only cycles with non-zero average power
            df_cycle = pd.DataFrame(cycle, columns=['Index', 'Value'])

            # Calculate time array
            df_length = len(df_cycle)
            time_increment = 1 / sample_rate
            time_array_length = df_length
            time_array = np.arange(0, time_array_length * time_increment, time_increment)

            # Ensure the length of the time array matches the length of DataFrame index
            if len(time_array) != df_length:
                if len(time_array) < df_length:
                    df_cycle = df_cycle.iloc[:len(time_array)]
                else:
                    df_cycle = df_cycle.reindex(range(len(time_array)), fill_value=0)

            df_cycle['Time'] = time_array
            df_cycle['Cycle'] = cycle_index + 1

            # Perform clustering
            max_index = df_cycle['Value'].idxmax()
            before_max = df_cycle.iloc[:max_index + 1]
            after_max = df_cycle.iloc[max_index:]

            X = after_max[['Value']]

            # Fit KMeans clustering algorithm
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(X)

            # Get cluster labels for after_max
            after_max['Cluster'] = kmeans.labels_

            # Assign the same cluster label to before_max
            before_max['Cluster'] = after_max['Cluster'].iloc[0]
            before_max['Status'] = 'rough'

            # Assign statuses based on the cluster order
            unique_clusters = after_max['Cluster'].unique()
            for cluster_index, cluster in enumerate(unique_clusters):
                if cluster_index == 0:
                    after_max.loc[after_max['Cluster'] == cluster, 'Status'] = 'rough'
                elif cluster_index == 1:
                    after_max.loc[after_max['Cluster'] == cluster, 'Status'] = 'semi finish'
                elif cluster_index == 2:
                    after_max.loc[after_max['Cluster'] == cluster, 'Status'] = 'finish'
                else:
                    after_max.loc[after_max['Cluster'] == cluster, 'Status'] = 'sparkout'

            # Merge before_max with after_max DataFrame
            merged_df = pd.concat([before_max, after_max])

            # Calculate status_time
            status_time = [0]
            for status in merged_df['Status'].unique():
                status_indices = merged_df.index[merged_df['Status'] == status]
                time_diff = merged_df.loc[status_indices[-1], 'Time'] - merged_df.loc[status_indices[0], 'Time']
                status_time.append(time_diff)
            status_time_df = pd.DataFrame({'Status_Time': status_time})

            # Append to lists
            final_df_list.append(merged_df)
            status_time_list.append(status_time_df)

    # Concatenate DataFrames
    final_df = pd.concat(final_df_list, ignore_index=True)
    status_time_df = pd.concat(status_time_list, ignore_index=True)

    return final_df, status_time_df


def generate_table(df):
    # Group by cycle and status
    grouped_df = df.groupby(['Cycle', 'Status'])

    # Calculate max and average value for each status in each cycle
    max_values = grouped_df['Value'].max().unstack()
    avg_values = grouped_df['Value'].mean().unstack()

    # Create table data
    table_data = []
    for cycle in df['Cycle'].unique():
        row_data = {
            'Cycle': cycle,
            'Rough (Max)': max_values.loc[cycle, 'rough'] if cycle in max_values.index else None,
            'Rough (Avg)': avg_values.loc[cycle, 'rough'] if cycle in avg_values.index else None,
            'Semi Finish (Max)': max_values.loc[cycle, 'semi finish'] if cycle in max_values.index else None,
            'Semi Finish (Avg)': avg_values.loc[cycle, 'semi finish'] if cycle in avg_values.index else None,
            'Finish (Max)': max_values.loc[cycle, 'finish'] if cycle in max_values.index else None,
            'Finish (Avg)': avg_values.loc[cycle, 'finish'] if cycle in avg_values.index else None,
            'Sparkout (Max)': max_values.loc[cycle, 'sparkout'] if cycle in max_values.index else None,
            'Sparkout (Avg)': avg_values.loc[cycle, 'sparkout'] if cycle in avg_values.index else None
        }
        table_data.append(row_data)

    # Convert the list of dictionaries to a DataFrame
    table_df = pd.DataFrame(table_data)

    # Transpose the DataFrame
    table_df = table_df.T
    print(table_df)

    # Manually set the header values to match the original order
    header_values = ['Cycle', 'Rough (Max)', 'Rough (Avg)', 'Semi Finish (Max)', 'Semi Finish (Avg)',
                     'Finish (Max)', 'Finish (Avg)', 'Sparkout (Max)', 'Sparkout (Avg)']

    # Create Plotly table
    table = go.Figure(data=[go.Table(
        header=dict(values=header_values,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=table_df.values.tolist(),
                   fill_color='lavender',
                   align='left'))
    ])
    table.update_layout(title_text='Part Analysis', title_x=0.5)

    return table, table_df


def genrate_cycle_part(df):
    return 'hi'


def adjust_signal(signal, ideal_value):
    """
    Adjusts the signal array by replacing values less than the ideal value with the ideal value.

    Parameters:
        signal (array_like): The signal array to be adjusted.
        ideal_value (float): The ideal value to compare against.

    Returns:
        adjusted_signal (array_like): The adjusted signal array.
    """
    adjusted_signal = [max(value, ideal_value) for value in signal]
    return adjusted_signal


def calculate_status_time(df, sample_rate=10, cycle_duration=20):
    # Sort the DataFrame by 'Cycle'
    df.sort_values(by=['Cycle', 'Time'], inplace=True)

    # Calculate cumulative time for each cycle
    df['CycleTime'] = df.groupby('Cycle').cumcount() / sample_rate * cycle_duration

    # Calculate time difference for each row
    df['TimeDiff'] = df.groupby(['Cycle', 'Status']).cumcount() / sample_rate

    # Calculate time taken for each status
    df['TimeTaken'] = df.groupby(['Cycle', 'Status'])['TimeDiff'].cumsum()

    # Get the last non-zero time taken for each status and cycle
    last_nonzero_time = df[df['TimeTaken'] != 0].groupby(['Cycle', 'Status'])['TimeTaken'].last().reset_index()

    return last_nonzero_time


@app.route('/')
def index():
    return render_template('login.html')

@app.route('/success')
def up():
    return render_template('tdms_upload.html')


@app.route('/upload', methods=['POST'])
def upload_folder():
    if 'folder' not in request.files:
        return 'No folder part'

    folder = request.files.getlist('folder')
    if not folder:
        return 'No selected folder'

    # Since only one file is uploaded, we'll take the first one
    uploaded_file = folder[0]

    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(file_path)

    data = read_tdms_file(file_path)
    if data is None:
        return f"Error reading TDMS file {file_path}"

    original_signal = data.get('Power', [])
    butter_3 = apply_butterworth_filter_order_3(original_signal)
    kalman = apply_kalman_filter(original_signal)
    butter_4 = apply_butterworth_filter_order_4(original_signal)
    savitzky = apply_savitzky_golay_filter(original_signal)
    gaussian = apply_gaussian_filter(original_signal)

    filtered_signal = apply_kalman_filter(original_signal)

    # Combine original_signal, filtered_signal, and time_data into a single DataFrame
    combined_df = pd.DataFrame({'Original_Signal': original_signal,
                                'Filtered_Signal': filtered_signal})

    ideal_power = get_min_value_matching_mode(filtered_signal)
    ideal_power_formatted_str = "{:.3f}".format(ideal_power)
    ideal_power_formatted_float = float(ideal_power_formatted_str)
    sig = adjust_signal(filtered_signal, ideal_power_formatted_float)

    time_df = time_generation(original_signal)
    time_df1 = time_generation(filtered_signal)
    total_time = time_df1['Time_Seconds'].iloc[-1]

    cycles = round(calculate_cycles(filtered_signal, ideal_power_formatted_float))
    cycle_times = calculate_cycle_times(filtered_signal, time_df, ideal_power_formatted_float)
    power = find_cycle_powers(filtered_signal, ideal_power_formatted_float)
    power_df = pd.DataFrame(power, columns=['Max_Power', 'Average_Power'])

    choose_the_best_filter = ['Butterworth_order 3', 'Butterworth_order 4',
                              'chebyshev_type1_order 3', 'chebyshev_type1_order_4', 'Kalman', 'savitzky_golay',
                              'Gaussian', 'Apply the Best Filter'
                              ]

    lms_4 = apply_lms_filter_order_4(original_signal)
    lms_3 = apply_lms_filter_order_3(original_signal)
    chebyshev_3 = apply_chebyshev_type1_filter_order_3(original_signal)
    chebyshev_4 = apply_chebyshev_type1_filter_order_4(original_signal)
    power_butt3 = find_cycle_powers(butter_3, ideal_power_formatted_float)
    power_df_b3 = pd.DataFrame(power_butt3, columns=['Max_Power', 'Average_Power'])
    power_butt4 = find_cycle_powers(butter_4, ideal_power_formatted_float)
    power_df_b4 = pd.DataFrame(power_butt4, columns=['Max_Power', 'Average_Power'])
    power_sav = find_cycle_powers(savitzky, ideal_power_formatted_float)
    power_df_sav = pd.DataFrame(power_sav, columns=['Max_Power', 'Average_Power'])
    power_gau = find_cycle_powers(gaussian, ideal_power_formatted_float)
    power_df_gau = pd.DataFrame(power_gau, columns=['Max_Power', 'Average_Power'])
    power_che3 = find_cycle_powers(chebyshev_3, ideal_power_formatted_float)
    power_df_c3 = pd.DataFrame(power_che3, columns=['Max_Power', 'Average_Power'])
    power_che4 = find_cycle_powers(chebyshev_4, ideal_power_formatted_float)
    power_df_c4 = pd.DataFrame(power_che4, columns=['Max_Power', 'Average_Power'])
    power_kal = find_cycle_powers(kalman, ideal_power_formatted_float)
    power_df_kal = pd.DataFrame(power_kal, columns=['Max_Power', 'Average_Power'])
    final_df, status_time_df = process_tdms_signal(sig, ideal_power_formatted_float)
    print(final_df)
    print(status_time_df)
    table_figure, table_part = generate_table(final_df)

    table_part_transposed = table_part.transpose()

    # Set the desired header values
    header_values = ['Cycle', 'Rough (Max)', 'Rough (Avg)', 'Semi Finish (Max)', 'Semi Finish (Avg)',
                     'Finish (Max)', 'Finish (Avg)', 'Sparkout (Max)', 'Sparkout (Avg)']

    # Set the columns of the transposed DataFrame to the header values
    table_part_transposed.columns = header_values

    filtered_df07 = table_part_transposed[table_part_transposed['Cycle'] == 1.0]
    print(table_part_transposed['Cycle'])

    # Select only the desired columns
    selected_columns = ['Rough (Avg)', 'Semi Finish (Avg)', 'Finish (Avg)']
    selected_data = filtered_df07[selected_columns]

    # Convert the selected data to a list
    selected_data_list = selected_data.values.tolist()

    sel1, sel2, sel3 = selected_data_list[0]

    # Create a DataFrame to store cycle number and time taken by each cycle
    cycle_data = pd.DataFrame(columns=['Cycle', 'Start_Time', 'End_Time', 'Duration'])

    # Iterate over cycle_times to calculate cycle durations and add them to the DataFrame
    for i, (start_time, end_time) in enumerate(cycle_times, 1):
        cycle_duration = end_time - start_time
        cycle_data = pd.concat([cycle_data, pd.DataFrame(
            {'Cycle': [i], 'Start_Time': [start_time], 'End_Time': [end_time], 'Duration': [cycle_duration]})],
                               ignore_index=True)

    # Save the DataFrame to a CSV file
    cycle_data.to_csv('cycle_data.csv', index=False)
    var = pd.DataFrame({'Cycle': cycle_data['Cycle'], 'Average_Power': power_df['Average_Power']})
    # idle power
    fig_idle = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ideal_power_formatted_float,
        title={'text': "Idle Power (kw)"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 2.5]},
            'steps': [
                {'range': [0, 4], 'color': "lightgray"}]  # Single color for the entire range
        }
    ))
    fig_cycle = go.Figure()

    # Add donut chart trace
    fig_cycle.add_trace(go.Pie(
        values=[cycles],
        hole=.3,
        marker_colors=['#66b3ff']))

    # Update layout
    fig_cycle.update_layout(title_text="Total Cycles: {}".format(cycles),
                            annotations=[dict(text=str(cycles),
                                              x=0.5, y=0.5, font_size=20, showarrow=False)])

    # Create a subplot with 1 row and 2 columns
    fig_original = go.Figure()
    fig_original.add_trace(go.Scatter(x=time_df['Time_Seconds'], y=original_signal,
                                      mode='lines', name='Original Signal'))
    fig_original.update_layout(title_text="Original Signal",
                               xaxis_title="Time (Seconds)",
                               yaxis_title="Power",
                               title_x=0.5)  # Set paper background color to black

    fig_filtered = go.Figure()
    fig_filtered.add_trace(go.Scatter(x=time_df['Time_Seconds'], y=sig,
                                      mode='lines', name='Filtered Signal'))
    fig_filtered.update_layout(title_text=f"Filtered Signal ",
                               # Use f-string to include the best filter name in the title
                               xaxis_title="Time (Seconds)",
                               yaxis_title="Power",
                               title_x=0.5)
    # Set paper background color to black
    fig_filtered_butter3 = go.Figure()
    fig_filtered_butter3.add_trace(go.Scatter(x=time_df['Time_Seconds'], y=butter_3,
                                              mode='lines', name='Filtered Signal'))
    fig_filtered_butter3.update_layout(title_text="Filtered Signal",
                                       xaxis_title="Time (Seconds)",
                                       yaxis_title="Power", title_x=0.5)
    fig_filtered_kalman = go.Figure()
    fig_filtered_kalman.add_trace(go.Scatter(x=time_df['Time_Seconds'], y=kalman,
                                             mode='lines', name='Filtered Signal'))
    fig_filtered_kalman.update_layout(title_text="Filtered Signal",
                                      xaxis_title="Time (Seconds)",
                                      yaxis_title="Power", title_x=0.5)
    fig_filtered_butter4 = go.Figure()
    fig_filtered_butter4.add_trace(go.Scatter(x=time_df1['Time_Seconds'], y=butter_4,
                                              mode='lines', name='Filtered Signal'))
    fig_filtered_butter4.update_layout(title_text="Filtered Signal",
                                       xaxis_title="Time (Seconds)",
                                       yaxis_title="Power", title_x=0.5)
    fig_filtered_cheb3 = go.Figure()
    fig_filtered_cheb3.add_trace(go.Scatter(x=time_df1['Time_Seconds'], y=chebyshev_3,
                                            mode='lines', name='Filtered Signal'))
    fig_filtered_cheb3.update_layout(title_text="Filtered Signal",
                                     xaxis_title="Time (Seconds)",
                                     yaxis_title="Power")
    fig_filtered_gaussian = go.Figure()
    fig_filtered_gaussian.add_trace(go.Scatter(x=time_df1['Time_Seconds'], y=gaussian,
                                               mode='lines', name='Filtered Signal'))
    fig_filtered_gaussian.update_layout(title_text="Filtered Signal",
                                        xaxis_title="Time (Seconds)",
                                        yaxis_title="Power")
    fig_filtered_cheb4 = go.Figure()

    fig_filtered_cheb4.add_trace(go.Scatter(x=time_df1['Time_Seconds'], y=chebyshev_4,
                                            mode='lines', name='Filtered Signal'))
    fig_filtered_cheb4.update_layout(title_text="Filtered Signal",
                                     xaxis_title="Time (Seconds)",
                                     yaxis_title="Power")

    fig_filtered_sav = go.Figure()
    fig_filtered_sav.add_trace(go.Scatter(x=time_df1['Time_Seconds'], y=savitzky,
                                          mode='lines', name='Filtered Signal'))
    fig_filtered_sav.update_layout(title_text="Filtered Signal",
                                   xaxis_title="Time (Seconds)",
                                   yaxis_title="Power (kw)")

    # Create pie chart for max power distribution
    fig_pie_max = go.Figure(go.Pie(labels=cycle_data['Cycle'], values=power_df['Max_Power'], name='Max Power'))
    fig_pie_max.update_layout(title="Max Power Distribution in Each Cycle",
                              )  # Set paper background color to black

    # Create pie chart for average power distribution
    fig_pie_avg = go.Figure(go.Pie(labels=cycle_data['Cycle'], values=power_df['Average_Power'], name='Average Power'))
    fig_pie_avg.update_layout(title="Average Power Distribution in Each Cycle",
                              )
    fig_pie_max_b3 = go.Figure(go.Pie(labels=cycle_data['Cycle'], values=power_df_b3['Max_Power'], name='Max Power'))
    fig_pie_max_b3.update_layout(title="Max Power Distribution in Each Cycle",
                                 )  # Set paper background color to black

    # Create pie chart for average power distribution
    fig_pie_avg_b3 = go.Figure(
        go.Pie(labels=cycle_data['Cycle'], values=power_df_b3['Average_Power'], name='Average Power'))
    fig_pie_avg_b3.update_layout(title="Average Power Distribution in Each Cycle",
                                 )  # Set paper background color to black
    fig_pie_max_b4 = go.Figure(go.Pie(labels=cycle_data['Cycle'], values=power_df_b4['Max_Power'], name='Max Power'))
    fig_pie_max_b4.update_layout(title="Max Power Distribution in Each Cycle",
                                 )  # Set paper background color to black

    # Create pie chart for average power distribution
    fig_pie_avg_b4 = go.Figure(
        go.Pie(labels=cycle_data['Cycle'], values=power_df_b4['Average_Power'], name='Average Power'))
    fig_pie_avg_b4.update_layout(title="Average Power Distribution in Each Cycle",
                                 )  # Set paper background color to black
    fig_pie_max_c3 = go.Figure(go.Pie(labels=cycle_data['Cycle'], values=power_df_c3['Max_Power'], name='Max Power'))
    fig_pie_max_c3.update_layout(title="Max Power Distribution in Each Cycle",
                                 )  # Set paper background color to black

    # Create pie chart for average power distribution
    fig_pie_avg_c3 = go.Figure(
        go.Pie(labels=cycle_data['Cycle'], values=power_df_c3['Average_Power'], name='Average Power'))
    fig_pie_avg_c3.update_layout(title="Average Power Distribution in Each Cycle",
                                 )  # Set paper background color to black
    fig_pie_max_c4 = go.Figure(go.Pie(labels=cycle_data['Cycle'], values=power_df_c4['Max_Power'], name='Max Power'))
    fig_pie_max_c4.update_layout(title="Max Power Distribution in Each Cycle",
                                 )  # Set paper background color to black

    # Create pie chart for average power distribution
    fig_pie_avg_c4 = go.Figure(
        go.Pie(labels=cycle_data['Cycle'], values=power_df_c4['Average_Power'], name='Average Power'))
    fig_pie_avg_c4.update_layout(title="Average Power Distribution in Each Cycle",
                                 )  # Set paper background color to black
    fig_pie_max_kal = go.Figure(go.Pie(labels=cycle_data['Cycle'], values=power_df_kal['Max_Power'], name='Max Power'))
    fig_pie_max_kal.update_layout(title="Max Power Distribution in Each Cycle",
                                  )  # Set paper background color to black

    # Create pie chart for average power distribution
    fig_pie_avg_kal = go.Figure(
        go.Pie(labels=cycle_data['Cycle'], values=power_df_kal['Average_Power'], name='Average Power'))
    fig_pie_avg_kal.update_layout(title="Average Power Distribution in Each Cycle",
                                  )  # Set paper background color to black
    fig_pie_max_sav = go.Figure(go.Pie(labels=cycle_data['Cycle'], values=power_df_sav['Max_Power'], name='Max Power'))
    fig_pie_max_sav.update_layout(title="Max Power Distribution in Each Cycle",
                                  )  # Set paper background color to black

    # Create pie chart for average power distribution
    fig_pie_avg_sav = go.Figure(
        go.Pie(labels=cycle_data['Cycle'], values=power_df_sav['Average_Power'], name='Average Power'))
    fig_pie_avg_sav.update_layout(title="Average Power Distribution in Each Cycle",
                                  )  # Set paper background color to black
    fig_pie_max_gas = go.Figure(go.Pie(labels=cycle_data['Cycle'], values=power_df_gau['Max_Power'], name='Max Power'))
    fig_pie_max_gas.update_layout(title="Max Power Distribution in Each Cycle",
                                  )  # Set paper background color to black

    # Create pie chart for average power distribution
    fig_pie_avg_gas = go.Figure(
        go.Pie(labels=cycle_data['Cycle'], values=power_df_gau['Average_Power'], name='Average Power'))
    fig_pie_avg_gas.update_layout(title="Average Power Distribution in Each Cycle",
                                  )  # Set paper background color to black
    # Set paper background color to black
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=cycle_data['Cycle'],  # Cycle numbers
        y=cycle_data['Duration'],  # Durations of each cycle
        marker=dict(color='green'),  # Set color for bars
        opacity=0.6  # Set opacity
    ))
    fig_bar.update_layout(
        title='Cycles',
        title_x=0.5 , # Set title
        xaxis_title='Cycle',  # Set x-axis label
        yaxis_title='Time(Seconds)',  # Set y-axis label
        bargap=0.2,  # Set gap between bars
        hovermode='closest',
        annotations = [
            dict(
                x=0.5,  # X coordinate of the annotation (centered)
                y=1.05,  # Y coordinate of the annotation (below the title)
                xref="paper",  # X reference of the annotation
                yref="paper",  # Y reference of the annotation
                text=f"Total Cycles {cycles}",  # Text to be displayed
                showarrow=False,  # Do not show arrow
                font=dict(  # Define font style
                    family="Arial",  # Font family
                    size=14,  # Font size
                    color="black"  # Font color
                ),
                align="center",  # Align text to the center
            )
        ]
    # Set hover mode
    )

    avg_power = power_df['Max_Power'].mean()

    # Create a list to store cycle numbers where power exceeds the average
    exceeding_cycles = []
    for i, power in enumerate(power_df['Max_Power']):
        try:
            if power > avg_power:
                exceeding_cycles.append(cycle_data['Cycle'][i])
        except KeyError:
            pass  # Skip if the key is not found

    # Define the text to display
    text = f"Cycles Exceeding Average Peak Power: {', '.join(map(str, exceeding_cycles)) if exceeding_cycles else 'None'}"

    # Create the chart
    fig = go.Figure()

    # Add scatter plot with lines
    fig.add_trace(go.Scatter(x=cycle_data['Cycle'], y=power_df['Max_Power'], mode='markers', name='Power'))

    # Add average power line
    fig.add_trace(go.Scatter(x=cycle_data['Cycle'], y=[avg_power] * len(cycle_data), mode='lines', name='Average Peak Power',
                             line=dict(color='red')))

    # Define layout
    fig.update_layout(
        title="Peak Power Trend",
        title_x=0.5,
        xaxis_title="Cycle",
        yaxis_title="Power",
        showlegend=True,
        annotations=[
            dict(
                x=1.02,  # Position the text in the right side free space
                y=1,
                text=text,
                align="left",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="black",
                borderwidth=1,
                font=dict(
                    size=12,
                    color="black"
                )
            )
        ]
    )
    avg_power_mean = power_df['Average_Power'].mean()

    # Create a list to store cycle numbers where power exceeds the average
    exceeding_cycles1 = []
    for i, power1 in enumerate(power_df['Average_Power']):
        try:
            if power1 > avg_power_mean:
                exceeding_cycles.append(cycle_data['Cycle'][i])
        except KeyError:
            pass  # Skip if the key is not found

    # Define the text to display
    #text = f"Cycles Exceeding Average Mean Power: {', '.join(map(str, exceeding_cycles1)) if exceeding_cycles1 else 'None'}"

    # Create the chart
    fig1 = go.Figure()

    # Add scatter plot with lines
    fig1.add_trace(go.Scatter(x=cycle_data['Cycle'], y=power_df['Average_Power'], mode='markers', name='Power'))

    # Add average power line
    fig1.add_trace(
        go.Scatter(x=cycle_data['Cycle'], y=[avg_power_mean] * len(cycle_data), mode='lines', name='Average Mean Power',
                   line=dict(color='red')))

    # Define layout
    fig1.update_layout(
        title="Mean Power Trend",
        title_x=0.5,
        xaxis_title="Cycle",
        yaxis_title="Power",
        showlegend=True,

    )
    fig_max = go.Figure()
    fig_max.add_trace(go.Bar(x=table_part_transposed['Cycle'], y=table_part_transposed['Rough (Max)'], name='Rough', marker_color='rgba(255, 0, 0, 0.6)' ,text=table_part_transposed['Rough (Max)'], textposition='inside'))
    fig_max.add_trace(go.Bar(x=table_part_transposed['Cycle'], y=table_part_transposed['Semi Finish (Max)'], name='Semi Finish', marker_color='rgba(0, 255, 0, 0.6)' ,text=table_part_transposed['Semi Finish (Max)'], textposition='inside'))
    fig_max.add_trace(go.Bar(x=table_part_transposed['Cycle'], y=table_part_transposed['Finish (Max)'], name='Finish', marker_color='rgba(0, 0, 255, 0.6)' ,text=table_part_transposed['Finish (Max)'], textposition='inside'))
    fig_max.add_trace(go.Bar(x=table_part_transposed['Cycle'], y=table_part_transposed['Sparkout (Max)'], name='Sparkout', marker_color='rgba(255, 165, 0, 0.6)' ,text=table_part_transposed['Sparkout (Max)'], textposition='inside'))

    fig_max.update_layout(barmode='stack', title='Max Power in each Grinding Stages', xaxis_title='Cycle',  yaxis_title='',yaxis=dict(visible=False),title_x =0.5)

    fig_avg = go.Figure()
    fig_avg.add_trace(go.Bar(x=table_part_transposed['Cycle'], y=table_part_transposed['Rough (Avg)'], name='Rough',
                             marker_color='rgba(255, 0, 0, 0.6)', text=table_part_transposed['Rough (Avg)'],
                             textposition='inside'))
    fig_avg.add_trace(
        go.Bar(x=table_part_transposed['Cycle'], y=table_part_transposed['Semi Finish (Avg)'], name='Semi Finish',
               marker_color='rgba(0, 255, 0, 0.6)', text=table_part_transposed['Semi Finish (Avg)'],
               textposition='inside'))
    fig_avg.add_trace(go.Bar(x=table_part_transposed['Cycle'], y=table_part_transposed['Finish (Avg)'], name='Finish',
                             marker_color='rgba(0, 0, 255, 0.6)', text=table_part_transposed['Finish (Avg)'],
                             textposition='inside'))
    fig_avg.add_trace(
        go.Bar(x=table_part_transposed['Cycle'], y=table_part_transposed['Sparkout (Avg)'], name='Sparkout',
               marker_color='rgba(255, 165, 0, 0.6)', text=table_part_transposed['Sparkout (Avg)'],
               textposition='inside'))

    fig_avg.update_layout(barmode='stack', title='Average Power in each Grinding Stages', xaxis_title='Cycle',
                          yaxis_title='', yaxis=dict(visible=False), title_x=0.5)

    # Convert figures to HTML
    fif_idle_html = pio.to_html(fig_idle)
    fig_cycle_html = pio.to_html(fig_cycle)
    fig_original_html = pio.to_html(fig_original)
    fig_filtered_html = pio.to_html(fig_filtered)
    fig_pie_max_html = pio.to_html(fig_pie_max)
    fig_pie_avg_html = pio.to_html(fig_pie_avg)
    fig_filtered_butter3_html = pio.to_html(fig_filtered_butter3)
    fig_filtered_kalman_html = pio.to_html(fig_filtered_kalman)
    fig_filtered_butter4_html = pio.to_html(fig_filtered_butter4)

    fig_filtered_gaussian_html = pio.to_html(fig_filtered_gaussian)
    fig_filtered_sav_html = pio.to_html(fig_filtered_sav)
    fig_filtered_cheb3_html = pio.to_html(fig_filtered_cheb3)
    fig_filtered_cheb4_html = pio.to_html(fig_filtered_cheb4)
    fig_pie_max_c3_html = pio.to_html(fig_pie_max_c3)
    fig_pie_avg_c3_html = pio.to_html(fig_pie_avg_c3)
    fig_pie_max_c4_html = pio.to_html(fig_pie_max_c4)
    fig_pie_avg_c4_html = pio.to_html(fig_pie_avg_c4)
    fig_pie_max_b3_html = pio.to_html(fig_pie_max_b3)
    fig_pie_avg_b3_html = pio.to_html(fig_pie_avg_b3)
    fig_pie_max_b4_html = pio.to_html(fig_pie_max_b4)
    fig_pie_avg_b4_html = pio.to_html(fig_pie_avg_b4)
    fig_pie_max_kal_html = pio.to_html(fig_pie_max_kal)
    fig_pie_avg_kal_html = pio.to_html(fig_pie_avg_kal)
    fig_pie_max_gas_html = pio.to_html(fig_pie_max_gas)
    fig_pie_avg_gas_html = pio.to_html(fig_pie_avg_gas)
    fig_pie_max_sav_html = pio.to_html(fig_pie_max_sav)
    fig_pie_avg_sav_html = pio.to_html(fig_pie_avg_sav)
    fig_bar_html = pio.to_html(fig_bar)
    table_html = pio.to_html(table_figure)
    peak_fig_html = pio.to_html(fig)
    mean_fig_html = pio.to_html(fig1)
    fig_max_html = pio.to_html(fig_max)
    fig_avg_html = pio.to_html(fig_avg)
    # Add annotations
    annotations_html = f"""
<div style='text-align: center;'>

\t\t<span style='font-family: sans-serif; color: black; font-size: 24px;'>Total Time Taken: {total_time} seconds</span>\t\t<br>
</div>
"""
    # Combine HTML strings into a single HTML layout
    dropdown_options = ''
    for cycle in cycle_data['Cycle']:
        dropdown_options += f"<option value='{cycle}'>Cycle {cycle}</option>"
    dropdown_options1 = ''
    for filter_option in choose_the_best_filter:
        dropdown_options1 += f"<option value='{filter_option}'>{filter_option}</option>"

    # Update HTML layout to include dropdown
    html_layout = f"""
    <!DOCTYPE html>
    <html>

    <head>
    <meta charset="UTF-8">
        <title>Grinding Analysis Dashboard</title>
        <style>
            body {{
                background-color: lightblue; /* Set background color to lightblue */
                color: black; /* Set text color to black */
                background-image: url('/static/dash.jpg');
                background-size: cover; /* Add background image */
                background-repeat: no-repeat; /* Do not repeat the background image */
                background-size: cover; /* Cover the entire body with the background image */
            }}
            .plot-container {{
                display: flex;
                justify-content: space-around;
                margin-bottom: 20px;
            }}
            .plot {{
                flex-basis: 45%;
            }}
            .input-container {{
            text-align: center;
            margin-bottom: 20px;}}
            h1 {{
                color: red;
                text-align: center;
                font-family: "sans-serif";
                font-size: 45px;
            }}
            .input-box {{
    border: 1px solid #ccc; /* Add a border around the box */
    padding: 10px; /* Add padding inside the box */
    border-radius: 5px; /* Add rounded corners */
    margin-bottom: 10px; /* Add some space between the box and the button */
}}
#show-mrr-column-btn {{
    background-color: red; /* Set the background color to red */
    color: white; /* Set the text color to white */
    padding: 10px 20px; /* Add padding to the button */
    font-size: 16px; /* Set the font size to medium */
    border: none; /* Remove the button border */
    border-radius: 5px; /* Add rounded corners */
    cursor: pointer; /* Change cursor to pointer on hover */
}}

#show-mrr-column-btn {{
    background-color: darkred; /* Change the background color on hover */
}}
#part-details {{
        font-family: sans-serif;
        font-size: 18px;
        text-align: center;
        border-collapse: collapse;
        margin: auto; /* Aligning the table in the center */
    }}
#power-vs-mrr {{
    background-color: red; /* Set the background color to red */
    color: white; /* Set the text color to white */
    padding: 10px 20px; /* Add padding to the button */
    font-size: 16px; /* Set the font size to medium */
    border: none; /* Remove the button border */
    border-radius: 5px; /* Add rounded corners */
    cursor: pointer; /* Change cursor to pointer on hover */
}}

#power-vs-mrr {{
    background-color: darkred; /* Change the background color on hover */
}}
#part-details {{
        font-family: sans-serif;
        font-size: 18px;
        text-align: center;
        border-collapse: collapse;
        margin: auto; /* Aligning the table in the center */
    }}
    #part-details th,
    #part-details td {{
        border: 1px solid black;
        padding: 8px;
    }}
    #cycle-display{{
    font-family: sans-serif;
        font-size: 18px;
        text-align: center;}}
    #save {{
    background-color: green; /* Set the background color to red */
    color: white; /* Set the text color to white */
    padding: 10px 20px; /* Add padding to the button */
    font-size: 16px; /* Set the font size to medium */
    border: none; /* Remove the button border */
    border-radius: 5px; /* Add rounded corners */
    cursor: pointer; /* Change cursor to pointer on hover */
}}

#save:hover {{
    background-color: darkred; /* Change the background color on hover */
}}
#ww{{
     font-family: sans-serif;
        font-size: 18px;}}
#thresholdPowerDiv
{{ font-family: sans-serif;
        font-size: 18px;
}}

        </style>
    </head>
    <body>
    <h1>Grinding Analysis Dashboard</h1>

    {annotations_html}
    <div class="plot-container">
    <div class="plot" id="idle">
            {fif_idle_html}
        </div>
     <div class="plot" id="cycle">
    
        {fig_bar_html}
    
        </div>
    </div>


    <!-- Dropdown list to display cycles -->
    <div style='text-align: center; margin-bottom: 20px;'>
    <span style='font-family: sans-serif; color: black; font-size: 24px;'>Select a Filter to Apply on your Signal </span>
    <br><br>
        <select id='Filter-dropdown' onchange='updateFilter(this.value)'>
            <option value='' disabled selected>Select a Filter</option>
            {dropdown_options1}
        </select>


    </div>
    <div id="threshold-power-container">  

    </div>



    <div class="plot-container">
        <div class="plot" id="plot1">
            {fig_original_html}
        </div>
        <div class="plot" id="plot2" style="display: none;">
            {fig_filtered_html}
        </div>
        <div class="plot" id="plot3" style="display: none;">
            {fig_filtered_butter3_html}
        </div>
        <div class="plot" id="plot4" style="display: none;">
            {fig_filtered_kalman_html}
        </div>
         <div class="plot" id="plot5" style="display: none;">
            {fig_filtered_cheb3_html}
        </div>
         <div class="plot" id="plot6" style="display: none;">
            {fig_filtered_cheb4_html}
        </div>
         <div class="plot" id="plot7" style="display: none;">
            {fig_filtered_sav_html}
        </div>
         <div class="plot" id="plot8" style="display: none;">
            {fig_filtered_gaussian_html}
        </div>
         <div class="plot" id="plot9" style="display: none;">
            {fig_filtered_butter4_html}
        </div>

    </div>

    <div class="plot-container">
        <div class="plot" id="plot11"  style="display: none;">
            {fig_pie_max_html}
        </div>
        <div class="plot" id="plot12"  style="display: none;">
            {fig_pie_avg_html}
        </div>
        <div class="plot" id="plot13"  style="display: none;">
            {fig_pie_max_b3_html}
        </div>
        <div class="plot" id="plot14"  style="display: none;">
            {fig_pie_avg_b3_html}
        </div>
        <div class="plot" id="plot15"  style="display: none;">
            {fig_pie_max_b4_html}
        </div>
        <div class="plot" id="plot16"  style="display: none;">
            {fig_pie_avg_b4_html}
        </div>
        <div class="plot" id="plot17"  style="display: none;">
            {fig_pie_max_c3_html}
        </div>
        <div class="plot" id="plot18"  style="display: none;">
            {fig_pie_avg_c3_html}
        </div>
        <div class="plot" id="plot19"  style="display: none;">
            {fig_pie_max_c4_html}
        </div>
        <div class="plot" id="plot20"  style="display: none;">
            {fig_pie_avg_c4_html}
        </div>
        <div class="plot" id="plot21"  style="display: none;">
            {fig_pie_max_kal_html}
        </div>
        <div class="plot" id="plot22"  style="display: none;">
            {fig_pie_avg_kal_html}
        </div>

        <div class="plot" id="plot24"  style="display: none;">
            {fig_pie_max_gas_html}
        </div>
        <div class="plot" id="plot25"  style="display: none;">
            {fig_pie_avg_gas_html}
        </div>
        <div class="plot" id="plot26"  style="display: none;">
            {fig_pie_max_sav_html}
        </div>
        <div class="plot" id="plot27"  style="display: none;">
            {fig_pie_avg_sav_html}
        </div>

    </div>
    
    <div class="plot-container">
    <div class="plot" id="bar-chart">
        {peak_fig_html}
    </div>
    <div class="plot" id="bar-chart">
    {mean_fig_html}
    </div>
    </div>
    <div class="plot-container">
    <div class="plot" id="bar-chart">
        {fig_max_html}
    </div>
    <div class="plot" id="bar-chart">
        {fig_avg_html}
    </div>
    </div>
    <div class="input-container">
    <div class="input-box">
        <label id="ww" for="wheel-width">Wheel Width in (mm):</label>
        <input type="number" id="wheelwidth"  name="ww" placeholder="Enter wheel width" oninput="updateWheelWidth(this)">


    </div>

    <p id="cycle-display"></p>
    <div>
    <table id="part-details">
        <caption>Part Details</caption>
        <thead>
            <tr>
                <th>Part</th>
                <th>Initial Diameter (mm)</th>
                <th>Final Diameter(mm)</th>
                <th class="mrr-column" style="display: none;">MRR(mm^3/sec)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Rough</td>
                <td><input type="number" class="part-input" placeholder="IR"></td>
                <td><input type="number" class="part-input" placeholder="FR"></td>
                <td class="mrr-column" style="display: none;"></td>
            </tr>
            <tr>
                <td>Semi-Finish</td>
                <td><input type="number" class="part-input" placeholder="Isf"></td>
                <td><input type="number" class="part-input" placeholder="Fsf"></td>
                <td class="mrr-column" style="display: none;"></td>
            </tr>
            <tr>
                <td>Finish</td>
                <td><input type="number" class="part-input" placeholder="If"></td>
                <td><input type="number" class="part-input" placeholder="Ff"></td>
                <td class="mrr-column" style="display: none;"></td>
            </tr>

        </tbody>
    </table>
    <br>

    <button id="show-mrr-column-btn" class="calculate-button">Calculate MRR</button>

</div>
<br>
<div id="thresholdPowerDiv" ></div><br>
<div class="plot-container">
<div  class ="plot" id="varsha"  ></div>

<div class ="plot" id="tp"></div>
</div>


<script>

        // Function to update plots based on selected cycle

        // Function to update plots based on selected filter
        var sel ={selected_data_list}
        function updateFilter(filter) {{
        console.log('Selected Filter:', filter);
        var plot2 = document.getElementById('plot2');
        var plot3 = document.getElementById('plot3');
        var plot4 = document.getElementById('plot4');
        var plot5 = document.getElementById('plot5');
        var plot6 = document.getElementById('plot6');
        var plot7 = document.getElementById('plot7');
        var plot8 = document.getElementById('plot8');
        var plot9 = document.getElementById('plot9');
        var plot10 = document.getElementById('plot10');
        var plot11 = document.getElementById('plot11');
        var plot12 = document.getElementById('plot12');
        var plot13 = document.getElementById('plot13');
        var plot14 = document.getElementById('plot14');
        var plot15 = document.getElementById('plot15');
        var plot16 = document.getElementById('plot16');
        var plot17 = document.getElementById('plot17');
        var plot18 = document.getElementById('plot18');
        var plot19 = document.getElementById('plot19');
        var plot20 = document.getElementById('plot20');
        var plot21 = document.getElementById('plot21');
        var plot22 = document.getElementById('plot22');

        var plot24 = document.getElementById('plot24');
        var plot25 = document.getElementById('plot25');
        var plot26 = document.getElementById('plot26');
        var plot27 = document.getElementById('plot27');

        plot2.style.display = 'none';
        plot3.style.display = 'none';
        plot4.style.display = 'none';
        plot5.style.display = 'none';
        plot6.style.display = 'none';
        plot7.style.display = 'none';
        plot8.style.display = 'none';
        plot9.style.display = 'none';
        plot11.style.display = 'none';
        plot12.style.display = 'none';
        plot13.style.display = 'none';
        plot14.style.display = 'none';
        plot15.style.display = 'none';
        plot16.style.display = 'none';
        plot17.style.display = 'none';
        plot18.style.display = 'none';
        plot19.style.display = 'none';
        plot20.style.display = 'none';
        plot21.style.display = 'none';
        plot22.style.display = 'none';

        plot24.style.display = 'none';
        plot25.style.display = 'none';
        plot26.style.display = 'none';
        plot27.style.display = 'none';


        if (filter === 'Butterworth_order 3') {{
            plot3.style.display = 'block';
            plot13.style.display = 'block';
            plot14.style.display = 'block';
        }} else if (filter === 'Kalman') {{
            plot4.style.display = 'block';
            plot21.style.display = 'block';
            plot22.style.display = 'block';
        }} else if (filter === 'Butterworth_order 4') {{
            plot9.style.display = 'block';
            plot15.style.display = 'block';
            plot16.style.display = 'block';
        }} else if (filter === 'chebyshev_type1_order 3') {{
            plot5.style.display = 'block';
            plot17.style.display = 'block';
            plot18.style.display = 'block';
        }} else if (filter === 'chebyshev_type1_order_4') {{
            plot6.style.display = 'block';
            plot19.style.display = 'block';
            plot20.style.display = 'block';
        }} else if (filter === 'savitzky_golay') {{
            plot7.style.display = 'block';
            plot26.style.display = 'block';
            plot27.style.display = 'block';
        }} else if (filter === 'Gaussian') {{
            plot8.style.display = 'block';
            plot24.style.display = 'block';
            plot25.style.display = 'block';
        }} else {{
            plot2.style.display = 'block';
            plot11.style.display = 'block';
            plot12.style.display = 'block';
        }}
    }}
  // Add event listener to the button to toggle visibility of MRR column
document.getElementById("show-mrr-column-btn").addEventListener("click", function() {{
    var mrrColumns = document.getElementsByClassName("mrr-column");
    for (var i = 0; i < mrrColumns.length; i++) {{
        mrrColumns[i].style.display = "table-cell";
    }}

    calculateMRR(); // Calculate MRR values
}});


// Function to update wheel width and recalculate MRR
function updateWheelWidth(input) {{
    var wheelwidth = parseFloat(input.value);
    calculateMRR()
}}




 function calculateMRR() {{
    var selec =[5.514,3.69,2.024];
    var mrrr =[20,10,7];
    var values ={selected_data_list};



    console.log(selec)

     // Your array of values

// Create an object with numerical keys representing the indices and assign the values
var select = {{}};
for (var i = 0; i < values.length; i++) {{
    select[i] = values[i];
}}
console.log(select)

// Add the length property to the object
select.length = values.length;


    // Get all rows in the table except the header row
    var rows = document.querySelectorAll("#part-details tbody tr");

    // Parse wheelwidth to float
    var wheelwidthFloat = parseFloat(document.getElementById("wheelwidth").value);
    console.log("Wheel Width:", wheelwidthFloat);


    // Define an array of divisor values for each row
    var divisorValues = [3, 0.9, 3.8];



var mrrValues = []; // Array to store MRR values

// Loop through each row
rows.forEach(function(row, rowIndex) {{
    // Get the input fields in the row
    var inputs = row.querySelectorAll("input[type='number']");
    if (inputs.length < 2) {{
        console.error("Row does not contain required input fields");
        return; // Skip this row if input fields are missing
    }}

    // Get the values from input fields
    var initialDiameter = parseFloat(inputs[0].value);
    var finalDiameter = parseFloat(inputs[1].value);

    // Check if input values are valid numbers
    if (isNaN(initialDiameter) || isNaN(finalDiameter)) {{
        console.error("Invalid input values in row:", row);
        return; // Skip this row if input values are not valid
    }}

    // Calculate radius difference
    var radius = initialDiameter - finalDiameter;

    // Calculate MRR values using corresponding divisor
    var divisor = divisorValues[rowIndex];
    var mrrValue = ((3.14 * radius)**2 * wheelwidthFloat / divisor).toFixed(2);

    // Log the calculated MRR value for debugging


    // Push MRR value to array
    mrrValues.push(parseFloat(mrrValue));

    // Get the corresponding cell for MRR value
    var mrrCell = row.querySelector(".mrr-column");

    // Update corresponding cell with the calculated MRR value
    mrrCell.style.display = ''; // Make the cell visible
    mrrCell.innerText = mrrValue; // Set the MRR value
}});

// Now, concatenate all MRR values into a single list
var allMRRValues = mrrValues.flat();
 // Using concat method
// or using spread operator
// var allMRRValues = [...mrrValues];

console.log(sel)

console.log("All MRR Values:", allMRRValues);
var xValues = allMRRValues;
var trace = {{
  x: allMRRValues,
  y: selec,
  type: 'scatter',
  mode: 'markers',
  name: 'Data' // Name of the plot (optional)
}};

// Calculate the mean of x and y values
var learningRate = 0.01;
var iterations = 1000;

// Calculate the mean of x and y values
var meanX = allMRRValues.reduce((acc, val) => acc + val, 0) / allMRRValues.length;
var meanY = selec.reduce((acc, val) => acc + val, 0) / select.length;

// Calculate the slope (m)
var numerator = 0;
var denominator = 0;
for (var i = 0; i < allMRRValues.length; i++) {{
  numerator += (allMRRValues[i] - meanX) * (selec[i] - meanY);
  denominator += Math.pow(allMRRValues[i] - meanX, 2);
}}
var slope = numerator / denominator;

// Initialize the intercept to the mean of y values
var intercept = meanY;

// Perform gradient descent
for (var i = 0; i < iterations; i++) {{
  // Calculate the gradient of the mean squared error with respect to the intercept
  var errorGradient = 0;
  for (var j = 0; j < allMRRValues.length; j++) {{
    var y_pred = slope * allMRRValues[j] + intercept;
    errorGradient += -2 * (selec[j] - y_pred);
  }}

  // Update the intercept
  intercept -= learningRate * (errorGradient / allMRRValues.length);
}}

// Compute y values for the regression line
var regressionYValues = allMRRValues.map(val => slope * val + intercept);
var squaredDifferences = [];
for (var i = 0; i < select.length; i++) {{
  var difference = regressionYValues[i] - selec[i];
  squaredDifferences.push(difference * difference);
}}

// Compute mean squared error
var sumSquaredDifferences = squaredDifferences.reduce((acc, val) => acc + val, 0);
var meanSquaredError = sumSquaredDifferences / select.length;

// Display mean squared error
console.log("Mean Squared Error:", meanSquaredError);

// Define your trace data for the regression line
var regressionTrace = {{
  x: allMRRValues,
  y: regressionYValues,
  type: 'scatter',
  mode: 'lines',

  name: 'Regression Line'
}};

// Plotly code to create the plot and show it (assuming Plotly is properly configured)
var trace = {{
  x: allMRRValues,
  y: selec,
  type: 'scatter',
  mode: 'markers',
  name: 'Data' // Name of the plot (optional)
}};

// Combine data and regression traces into an array
var dataWithRegression = [trace, regressionTrace];

// Plotly layout options
var layout = {{
  title: 'Power vs MRR',
  xaxis: {{
    title: 'MRR (mm^3)/sec',
    range: [0, Math.max(...allMRRValues)]
  }},
  yaxis: {{
    title: 'Power',
    range: [0, Math.max(...selec, ...regressionYValues)] // Adjusted to include regression line
  }}
}};
// Plot the graph
Plotly.newPlot('varsha', dataWithRegression, layout);
var thresholdDiv = document.getElementById("thresholdPowerDiv");
thresholdDiv.textContent = "Threshold Power: " + intercept.toFixed(2) + " (kW)";

var thresholdPower = intercept.toFixed(2);
console.log(thresholdPower);
var data = [
    {{
      domain: {{ x: [0, 1], y: [0, 1] }},
      value: thresholdPower,
      title: {{ text: "Threshold Power (kW)" }},
      type: "indicator",
      mode: "gauge+number",
      gauge: {{
        axis: {{ range: [null, 10] }},
        bar: {{ color: "darkblue" }},
        bgcolor: "white",
        borderwidth: 2,
        bordercolor: "gray",

        threshold: {{
          line: {{ color: "red", width: 4 }},
          thickness: 0.75,
          value: thresholdPower
        }}
      }}
    }}
  ];


Plotly.newPlot('tp', data);
}};
// Combine trace and layout, and plot the graph

// Call calculateMRR when the button is clicked

    </script>

    </body>
    </html>
    """


    return html_layout





if __name__ == '__main__':
    app.run(debug=True)