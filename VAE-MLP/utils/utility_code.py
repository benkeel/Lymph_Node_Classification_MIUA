import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from collections import Counter

def plot_results(results_path, save_results_path, filename):   #train_losses, test_losses
    '''
    method for plotting accuracy and loss graphs and saving file in given destination
    Input:
    - results_path: where to save the error plot
    - filename: name given to loss graph file
    '''
    data = torch.load(save_results_path)
    loss = data["train_losses"]
    val_loss = data["test_losses"]
    loss = loss[1:]
    val_loss = val_loss[1:]
    fig, ax1 = plt.subplots()
    plt.plot(loss, 'm', label = 'training loss')
    plt.plot(val_loss, 'g', label = 'test loss')
    plt.yscale("log")
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    fig.savefig(os.path.join(results_path, filename))
    plt.show()
    plt.close()

# add loss graph
def plot_MLP_results(training_results, hyperparams, results_path=None, filename=None):
    '''
    method for plotting AUC, sensitivity and loss graphs then saving file in given destination
    Input:
    - results_path: where to save the error plot
    - filename: name given to loss graph file
    '''
    hyperparams_str = '\n'.join([f'{key}: {value}' for key, value in hyperparams.items()])
    # Plotting the training results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Plot losses
    epochs = range(1, len(training_results['train_losses']) + 1)
    ax1.plot(epochs, training_results['train_losses'], label='Train Loss', color='blue')
    ax1.plot(epochs, training_results['test_losses'], label='Test Loss', color='orange')
    ax1.set_title('Train and Test Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot AUCs
    ax2.plot(epochs, training_results['train_AUCs'], label='Train AUC', color='green')
    ax2.plot(epochs, training_results['test_AUCs'], label='Test AUC', color='red')
    ax2.set_title('Train and Test AUC per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.legend()
    ax2.grid(True)

    # Plot Sensitivities
    ax3.plot(epochs, training_results['train_sensitivitys'], label='Train Sensitivity', color='purple')
    ax3.plot(epochs, training_results['test_sensitivitys'], label='Test Sensitivity', color='brown')
    ax3.set_title('Train and Test Sensitivity per Epoch')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Sensitivity')
    ax3.legend()
    ax3.grid(True)

    plt.gcf().text(0.75, 0.75, f'Hyperparameters:\n{hyperparams_str}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()

    # Save and show the plot
    if filename is not None:
        fig.savefig(os.path.join(results_path, filename))
    plt.show()
    plt.close()
    print('Hyperparameters: ', hyperparams)

# Compute number of parameters in VAE model
def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# find bounding boxes and centres for each mask

# find tight bounding box around each lymph node
def find_centers_and_bounding_boxes(mask):
    # Find unique object labels, ignoring the background (label 0)
    object_labels = np.unique(mask)
    object_labels = object_labels[object_labels != 0]

    centers_dict = {}
    bounding_boxes_dict = {}

    for label in object_labels:
        # Find the indices of the current object
        object_indices = np.argwhere(mask == label)

        # Calculate the center of the object
        center = object_indices.mean(axis=0).astype(int)
        centers_dict[label] = tuple(center)

        # Calculate the bounding box
        min_indices = object_indices.min(axis=0)
        max_indices = object_indices.max(axis=0)
        size = max_indices - min_indices + 1  # +1 because both ends are inclusive

        bounding_boxes_dict[label] = {
            "center": tuple(center),
            "size": tuple(size),
            "bounding_box": (tuple(min_indices), tuple(max_indices))
        }

    return bounding_boxes_dict

# find 32x32 bounding box around each slice of each lymph node
def find_centers_and_bounding_boxes_32(mask):
    # Find unique object labels, ignoring the background (label 0)
    object_labels = np.unique(mask)
    object_labels = object_labels[object_labels != 0]

    centers_dict = {}
    bounding_boxes_dict = {}

    for label in object_labels:
        # Find the indices of the current object
        object_indices = np.argwhere(mask == label)

        # Group indices by the first dimension (slice index)
        slices = np.unique(object_indices[:, 0])

        for slice_idx in slices:
            slice_indices = object_indices[object_indices[:, 0] == slice_idx]

            # Calculate the center of the object in this slice
            center = slice_indices.mean(axis=0).astype(int)
            center[0] = slice_idx  # Ensure the slice index is correctly set
            centers_dict[(label, slice_idx)] = tuple(center)

            # Define the bounding box limits
            x_center, y_center = center[1], center[2]
            x_min = max(x_center - 16, 0)
            x_max = min(x_center + 15, mask.shape[1] - 1)
            y_min = max(y_center - 16, 0)
            y_max = min(y_center + 15, mask.shape[2] - 1)

            bounding_boxes_dict[(label, slice_idx)] = {
                "center": tuple(center),
                "bounding_box": ((slice_idx, x_min, y_min), (slice_idx, x_max, y_max))
            }

            #assert (1, x_max - x_min + 1, y_max - y_min + 1) == (1, 32, 32)

    return bounding_boxes_dict #centers_dict,

def find_bounding_cubes(mask):
    # Find mask size
    mask_size = mask.shape
    # Find unique object labels, ignoring the background (label 0)
    object_labels = np.unique(mask)
    object_labels = object_labels[object_labels != 0]

    centers_dict = {}
    bounding_boxes_dict = {}

    half_size = np.array([2, 18, 18])  # Half of the bounding box size [4, 32, 32]/2
    bounding_cube_size = (4, 32, 32)

    for label in object_labels:
        # Find the indices of the current object
        object_indices = np.argwhere(mask == label)

        # Calculate the center of the object
        center = object_indices.mean(axis=0).astype(int)
        centers_dict[label] = tuple(center)

        # Calculate tight bounding cube
        min_indices = object_indices.min(axis=0)
        max_indices = object_indices.max(axis=0)

        # Calculate start and end indices for the bounding cube
        start_indices = center - half_size
        end_indices = center + half_size

        # Ensure the indices are within the image bounds
        start_indices = np.maximum(start_indices, 0)
        end_indices = np.minimum(end_indices, np.array(mask.shape) - 1)

        # Adjust for dimensions that may have different sizes
        start_indices = np.minimum(start_indices, np.array(mask.shape) - bounding_cube_size)
        end_indices = start_indices + bounding_cube_size

        # Extract the bounding cube
        slices = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))
        bounding_cube = mask[slices]

        # check if any indicies are negative or greater than the mask size
        if np.any(start_indices < 0) or np.any(end_indices > mask_size):
            print(f"Warning: Bounding cube for label {label} is out of bounds")
            print(start_indices, end_indices, mask_size)


        # Store the bounding cube
        bounding_boxes_dict[label] = {
            "center": tuple(center),
            "bounding_cube": (tuple(start_indices), tuple(end_indices)),
            "tight_bounding_cube": (tuple(min_indices), tuple(max_indices))
        }

    return bounding_boxes_dict

def get_single_scan_file_list(all_files_list, IMAGE_DIR, cohort1):
    # get ids of patients
    patient_files_list = [f for f in os.listdir(IMAGE_DIR + '\mri')] + [f  for f in os.listdir(IMAGE_DIR + '\mri_aug')]
    patient_files_list.sort()
    patient_ids = []
    for i in range(len(patient_files_list)):
        patient_id = patient_files_list[i][:10]
        if patient_id in cohort1['shortpatpseudoid'].values:
            if patient_id not in patient_ids:
                patient_ids.append(patient_files_list[i][:10])
    patient_ids.sort()
    # only one scan per patient to avoid repeated LNs
    all_files_list2 = []
    for pat_id in patient_ids:
        scan_ids = {}
        for file in all_files_list:
            if pat_id in file:
                scan_id = file.split('//')[1][:31]
                if scan_id not in scan_ids:
                    scan_ids[scan_id] = [file]
                else:
                    scan_ids[scan_id] += [file]
        lens = [len(scan_ids[key]) for key in scan_ids]
        if len(lens) == 1:
            all_files_list2 += scan_ids[list(scan_ids.keys())[0]]
        else:
            max_len = max(lens)
            for key in scan_ids:
                if len(scan_ids[key]) == max_len:
                    all_files_list2 += scan_ids[key]
                    break # only one scan per patient (8E8F6CEC43 has two max of same number of files)
    return all_files_list2


def get_class_distribution(ids, train_test, patient_labels_dict):
    p, n = 0, 0
    for id in ids:
        if patient_labels_dict[id] == 0:
            n += 1
        else:
            p += 1
    print(f'{train_test}: {n+p}, Positive samples: {p}, Negative samples: {n}, Positive proportion: {p/(n+p):.2f}')
    return "Negative ratio:", n/p

def weights_init(m, backbone_indicator=False):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    if not backbone_indicator:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

def error_analysis(probabilities_list, labels, results_path, threshold=0.5, fold_idx=-42):
    # find elementwise average across probabilities
    print('Average confidence:', np.mean(probabilities_list, axis=0))
    correct_positive_indices_list, incorrect_positive_indices_list, correct_negative_indices_list, incorrect_negative_indices_list = [], [], [], []

    for i in range(len(probabilities_list)):
        probabilities = probabilities_list[i]
        predictions = np.array([1 if j >= threshold else 0 for j in probabilities])

        # Correct positive predictions
        correct_positive_indices = np.where((labels == 1) & (predictions == 1))[0]

        # Incorrect positive predictions
        incorrect_positive_indices = np.where((labels == 0) & (predictions == 1))[0]

        # Confidence of positive predictions
        positive_predictions_probabilities = probabilities[predictions == 1]

        # print('Positive')
        # print('Correct positive indices and confidence:', correct_positive_indices, probabilities[correct_positive_indices])
        # print('Incorrect positive indices and confidence:', incorrect_positive_indices, probabilities[incorrect_positive_indices])

        # Correct positive predictions
        correct_negative_indices = np.where((labels == 0) & (predictions == 0))[0]

        # Incorrect positive predictions
        incorrect_negative_indices = np.where((labels == 1) & (predictions == 0))[0]

        # Confidence of positive predictions
        negative_predictions_probabilities = probabilities[predictions == 0]

        # print('Negative')
        # print('Correct negative indices and confidence:', correct_negative_indices, probabilities[correct_negative_indices])
        # print('Incorrect negative indices and confidence:', incorrect_negative_indices, probabilities[incorrect_negative_indices])

        correct_positive_indices_list.extend(correct_positive_indices)
        incorrect_positive_indices_list.extend(incorrect_positive_indices)
        correct_negative_indices_list.extend(correct_negative_indices)
        incorrect_negative_indices_list.extend(incorrect_negative_indices)

    correct_positive_counter = Counter(correct_positive_indices_list)
    incorrect_positive_counter = Counter(incorrect_positive_indices_list)
    correct_negative_counter = Counter(correct_negative_indices_list)
    incorrect_negative_counter = Counter(incorrect_negative_indices_list)

    average_confidence = np.mean(probabilities_list, axis=0)

    fig, ax = plt.subplots(4, 1, figsize=(12, 16))
    # Correct Positive Predictions
    ax[0].bar(correct_positive_counter.keys(), correct_positive_counter.values(), color='green')
    ax[0].set_title('Frequency of Correct Positive (TP) Predictions Indices Over Multiple Runs')
    ax[0].set_xlabel('Index')
    ax[0].set_ylabel('Frequency')
    ax[0].set_xticks(list(correct_positive_counter.keys()))

    # Incorrect Negative Predictions
    ax[1].bar(incorrect_negative_counter.keys(), incorrect_negative_counter.values(), color='red')
    ax[1].set_title('Frequency of Incorrect Negative (FN) Predictions Indices Over Multiple Runs')
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('Frequency')
    ax[1].set_xticks(list(incorrect_negative_counter.keys()))

    # Incorrect Positive Predictions
    ax[2].bar(incorrect_positive_counter.keys(), incorrect_positive_counter.values(), color='red')
    ax[2].set_title('Frequency of Incorrect Positive (FP) Predictions Indices Over Multiple Runs')
    ax[2].set_xlabel('Index')
    ax[2].set_ylabel('Frequency')
    ax[2].set_xticks(list(incorrect_positive_counter.keys()))

    # Correct Negative Predictions
    ax[3].bar(correct_negative_counter.keys(), correct_negative_counter.values(), color='green')
    ax[3].set_title('Frequency of Correct Negative (TN) Predictions Indices Over Multiple Runs')
    ax[3].set_xlabel('Index')
    ax[3].set_ylabel('Frequency')
    ax[3].set_xticks(list(correct_negative_counter.keys()))
    plt.tight_layout()
    if fold_idx > -1:
        plt.savefig(results_path + f"/error_analysis_fold_{fold_idx}.png")
    else:
        plt.savefig(results_path + "/error_analysis.png")
    plt.show()





