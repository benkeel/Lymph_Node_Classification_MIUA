import torch
import torch.nn.functional as F
from torchvision import transforms
import nibabel as nib
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from utils.utility_code import get_single_scan_file_list, get_class_distribution


# Create class to load image data from files
class LoadImages():
    def __init__(self, main_dir, files_list):
        # Set the loading directory
        self.main_dir = main_dir
        # Transforms
        self.transform = transforms.Compose([transforms.ToTensor()])
        # Get list of all image file names
        self.all_imgs = files_list


    def __len__(self):
        # Return the previously computed number of images
        return len(self.all_imgs)

    def __getitem__(self, index):
        # Get image location
        img_loc = self.main_dir + self.all_imgs[index]
        # Represent image as a tensor
        img = nib.load(img_loc)
        img = img.get_fdata()
        img = self.transform(img)
        return img


class Load_Latent_Vectors():
    def __init__(self, patient_slices, latent_vectors, patient_labels, patient_ids, cohort1, all_files_list, short_long_axes_dict, mask_sizes, clinical_data_options, max_nodes=25, add_images=False):
        self.patient_ids = patient_ids
        self.patient_slices = patient_slices
        self.latent_vectors = latent_vectors
        self.patient_labels = patient_labels
        self.cohort1 = cohort1
        self.all_files_list = all_files_list
        self.short_long_axes_dict = short_long_axes_dict
        self.mask_sizes = mask_sizes
        self.max_nodes = max_nodes
        self.clinical_data_options = clinical_data_options

        self.add_images = add_images
        if add_images == True:
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.img_dir = r"C:\Users\mm17b2k.DS\Documents\ARCANE_Data\Cohort1_2D_slices/"


        with open(r"C:\Users\mm17b2k.DS\Documents\ARCANE_Data\border_metrics.pkl", 'rb') as f:
            compactness, convexity = pickle.load(f)

        self.compactness = compactness
        self.convexity = convexity

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        indices = self.patient_slices[patient_id]
        if len(indices) > self.max_nodes:
            mask_sizes = self.mask_sizes[patient_id]
            sizes = sorted(enumerate(mask_sizes), key=lambda x: x[1], reverse=True)
            biggest_n_mask_idx = [i for i, size in sizes[:self.max_nodes]]
            indices = sorted([indices[i] for i in biggest_n_mask_idx])

        # add clinical data to the latent vectors
        patient_indicator = False
        patient_options = []
        if "patient" in self.clinical_data_options:
            patient_indicator = True
            patient_options.append('age_scaled')
            patient_options.append('sex_numeric')
        if "T_stage" in self.clinical_data_options:
            patient_indicator = True
            patient_options.append('TumourLabel_numeric')

        if patient_options == []:
            patient_clinical_data = None
        else:
            patient_clinical_data = self.cohort1[self.cohort1['shortpatpseudoid'] == patient_id.split('_')[0]][patient_options].values.tolist()

        LN_features = []
        images = np.zeros((len(indices), 1, 32, 32))
        for i, index in enumerate(indices):
            file_name = self.all_files_list[index]
            pat_id = patient_id.split('_')[0]
            node_number = float(file_name.split('//')[1].split('_')[6])
            # print(file_name, node_number)
            long, short, ratio = self.short_long_axes_dict[pat_id][node_number]
            mask_file = file_name.replace('mri', 'mask')
            compactness = self.compactness[mask_file]
            convexity = self.convexity[mask_file]

            node_indicator = False
            node_options = []
            if "size" in self.clinical_data_options:
                node_indicator = True
                node_options.append(long)
                node_options.append(short)
                node_options.append(ratio)
            if "border" in self.clinical_data_options:
                node_indicator = True
                node_options.append(compactness)
                node_options.append(convexity)

            LN_features.append(node_options)

            if self.add_images == True:
                img = nib.load(self.img_dir + file_name).get_fdata()
                img = self.transform(img)
                # print(img.shape, i, index)
                images[i] = img


            #clinical_features.append([long, short, ratio, compactness, convexity]) #, patient_clinical_data[0][0], patient_clinical_data[0][1], patient_clinical_data[0][2]])



        LN_features = np.array(LN_features)
        if patient_indicator == True and ("size" in self.clinical_data_options):
            patient_clinical_data = patient_clinical_data[0] + LN_features[np.argmax(LN_features[:, 0])].tolist() # add the data for node with max long axis ratio (and corresponding short/ratio/compactness/convexity) to patient level clinical data
        if patient_indicator == True and ("border" in self.clinical_data_options) and ("size" not in self.clinical_data_options):
            patient_clinical_data = patient_clinical_data[0] + LN_features[np.argmin(LN_features[:, 0])].tolist() # add min compactness node data to patient level clinical data
        if patient_indicator == True and node_indicator == False:
            patient_clinical_data = patient_clinical_data[0]
        if patient_indicator == False and node_indicator == True:
            if "size" in self.clinical_data_options:
                patient_clinical_data = LN_features[np.argmax(LN_features[:, 0])].tolist()
            if ("border" in self.clinical_data_options) and ("size" not in self.clinical_data_options):
                patient_clinical_data = LN_features[np.argmin(LN_features[:, 0])].tolist()
        if patient_indicator == False and node_indicator == False:
            patient_clinical_data = []

        label = self.patient_labels[patient_id]

        if self.add_images == True:
            number_of_nodes = len(images)
            if len(images) < self.max_nodes:
                #print(LN_features.shape, label, patient_clinical_data, number_of_nodes)
                images = np.concatenate((images, np.zeros((self.max_nodes - len(images), 1, 32, 32))), axis=0)
                LN_features = np.concatenate((LN_features, np.ones((self.max_nodes - len(LN_features), LN_features.shape[1]))*0.5), axis=0)
                #print(images.shape, LN_features.shape)

            return torch.tensor(LN_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long), torch.tensor(patient_clinical_data, dtype=torch.float32), torch.tensor(number_of_nodes, dtype=torch.float32), torch.tensor(images, dtype=torch.float32)


        if self.add_images == False:
            features = self.latent_vectors[indices]
            number_of_nodes = len(features)
            if ("size" in self.clinical_data_options) or ("border" in self.clinical_data_options):
                features = np.concatenate((features, LN_features), axis=1)

            if len(features) < self.max_nodes:
                features = np.concatenate((features, np.zeros((self.max_nodes - len(features), features.shape[1]))), axis=0)

            return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long), torch.tensor(patient_clinical_data, dtype=torch.float32), torch.tensor(number_of_nodes, dtype=torch.float32)




def add_data_for_pat_id(pat_id, cohort1, all_files_list, files_to_skip, patient_slices_dict, patient_labels_dict, patient_file_names_dict, mask_sizes, mask_sizes_patients, labels, num_synthetic_patients):
    # get all the file names for the patient (excluding any which are to be skipped - see prepare_VAE_MLP_joint_data() for file_to_skip)
    patient_file_names_id, patient_file_names = [], []
    for i in range(len(all_files_list)):
        if all_files_list[i].split('//')[1][:10] == pat_id:
            if all_files_list[i] not in files_to_skip:
                patient_file_names_id.append([all_files_list[i], i])
                patient_file_names.append(all_files_list[i])

    patient_file_names_dict[pat_id] = patient_file_names

    # collect slices for original patient
    patient_slices = []
    #patient_filtered_files = []
    node_numbers = []
    for file, j in patient_file_names_id:
        if 'aug' not in file:
            patient_slices.append(j)
            #patient_filtered_files.append(file)
            if file.split('_')[-2] not in node_numbers:
                node_numbers.append(file.split('_')[-2])

            patient_id = file.split('//')[1].split('_')[0]
            if patient_id not in mask_sizes_patients.keys():
                mask_sizes_patients[patient_id] = [mask_sizes[file.replace('mri', 'mask')]]
            else:
                mask_sizes_patients[patient_id].append(mask_sizes[file.replace('mri', 'mask')])

    patient_slices_dict[pat_id] = patient_slices

    # if only using non-augmented files to retrain VAE? check this/add functionality for if VAE retrain
    #patient_file_names_dict[pat_id] = patient_filtered_files

    N = cohort1[cohort1[('shortpatpseudoid')] == pat_id]['NodeLabel'].item()
    if N == '0':
        patient_labels_dict[pat_id] = 0
        labels.append(0)
    else:
        patient_labels_dict[pat_id] = 1
        labels.append(1)


    # add synthetic patients with augmented representations of each node
    for k in range(int(num_synthetic_patients)):
        patient_slices = []
        for node in node_numbers:
            done = False
            attempt = 0
            while done == False:
                attempt += 1
                if attempt == 100:
                    print('Could not find augmented scan for node', node, 'for patient', pat_id)
                    break
                aug_idx = str(np.random.randint(0, 25)) # choose one of the 25 augmented scans for each node
                for file, l in patient_file_names_id:
                    # augmented file that is for the specific node (can be multiple per node)
                    if 'aug' in file and node == file.split('_')[-3] and aug_idx == file.split('_')[-1][:-4]:
                        if file not in files_to_skip:
                            patient_slices.append(l)
                            patient_id = '_'.join([file.split('//')[1].split('_')[0], ''.join(['aug', str(k+1)])])
                            if patient_id not in mask_sizes_patients.keys():
                                mask_sizes_patients[patient_id] = [mask_sizes[file.replace('mri', 'mask')]]
                            else:
                                mask_sizes_patients[patient_id].append(mask_sizes[file.replace('mri', 'mask')])

                if patient_slices != []:
                    #print(len(patient_slices), [all_files_list[i] for i in patient_slices])
                    done = True

        patient_slices_dict[pat_id + '_aug' + str(k+1)] = patient_slices
        patient_file_names_dict[pat_id + '_aug' + str(k+1)] = patient_file_names
        N = cohort1[cohort1[('shortpatpseudoid')] == pat_id]['NodeLabel'].item()
        if N == '0':
            patient_labels_dict[pat_id + '_aug' + str(k+1)] = 0
            labels.append(0)
        else:
            patient_labels_dict[pat_id + '_aug' + str(k+1)] = 1
            labels.append(1)

    return patient_slices_dict, patient_labels_dict, patient_file_names_dict, mask_sizes_patients, labels

def augment_data(cohort1, all_files_list, files_to_skip, patient_ids, mask_sizes, num_synthetic, oversample_ratio):
    # get latent vector idx for each patient and add 10 synthetic patients
    patient_slices_dict = {}
    patient_labels_dict = {}
    patient_file_names_dict = {}
    mask_sizes_patients = {}
    labels = []

    for pat_id in patient_ids:
        N = cohort1[cohort1[('shortpatpseudoid')] == pat_id]['NodeLabel'].item()
        if N == '0':
            patient_slices_dict, patient_labels_dict, patient_file_names_dict, mask_sizes_patients, labels = add_data_for_pat_id(pat_id, cohort1, all_files_list, files_to_skip, patient_slices_dict, patient_labels_dict, patient_file_names_dict, mask_sizes, mask_sizes_patients, labels, num_synthetic_patients=num_synthetic)

        else:
            patient_slices_dict, patient_labels_dict, patient_file_names_dict, mask_sizes_patients, labels = add_data_for_pat_id(pat_id, cohort1, all_files_list, files_to_skip, patient_slices_dict, patient_labels_dict, patient_file_names_dict, mask_sizes, mask_sizes_patients, labels, num_synthetic_patients=num_synthetic*oversample_ratio) # oversample positive cases

    # get patient ids for full synthetic population
    synthetic_patient_ids = list(patient_labels_dict.keys())
    assert len(synthetic_patient_ids) == len(labels) == len(list(patient_slices_dict.keys()))

    return patient_slices_dict, patient_labels_dict, patient_file_names_dict, mask_sizes_patients, labels


def prepare_VAE_MLP_joint_data(first_time_train_test_split=True, cross_val=False, fold_data=None, train_ids=None, test_ids=None, num_synthetic=10, oversample_ratio=1):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    IMAGE_DIR = r"C:\Users\mm17b2k.DS\Documents\ARCANE_Data\Cohort1_2D_slices"

    cohort1 = pd.read_excel(r"C:\Users\mm17b2k.DS\Documents\ARCANE_Data\Cohort1.xlsx")

    all_files_list = ['\mri' + '//' + f for f in os.listdir(IMAGE_DIR + '\mri')] + ['\mri_aug' + '//' + f  for f in os.listdir(IMAGE_DIR + '\mri_aug')]
    all_files_list.sort()
    all_files_list = get_single_scan_file_list(all_files_list, IMAGE_DIR, cohort1)

    # get mask sizes
    mask_files_list = ['\mask' + '//' + f for f in os.listdir(IMAGE_DIR + '\mask')] + ['\mask_aug' + '//' + f  for f in os.listdir(IMAGE_DIR + '\mask_aug')]
    mask_sizes = {}
    for file in mask_files_list:
        mask = nib.load(IMAGE_DIR + file).get_fdata()
        mask_sizes[file] = np.sum(mask)



    patient_node_slice_sizes = {}
    for file in mask_files_list:
        mask_size = mask_sizes[file]
        if 'mask_aug' in file:
            patient_node_name = '_'.join(['_'.join(file.split('_')[:8]), file.split('_')[-1].split('.')[0]])
        else:
            patient_node_name = '_'.join(file.split('_')[:7])

        file = file.replace('mask', 'mri')

        if patient_node_name in patient_node_slice_sizes.keys():
            patient_node_slice_sizes[patient_node_name].append([mask_size, file])
        else:
            patient_node_slice_sizes[patient_node_name] = [[mask_size, file]]

    files_to_skip = []
    non_max_total = []
    for node in patient_node_slice_sizes.keys():
        if len(patient_node_slice_sizes[node]) > 1:
            #print(node, len(patient_node_slice_sizes[node]))
            sizes = [item[0] for item in patient_node_slice_sizes[node]]

            max_idx = sizes.index(max(sizes))
            max_size = sizes[max_idx]

            # delete other slice for same node if it is less than 50% of max size
            threshold_size = max_size * 0.50

            # get index of all non_max
            non_max_idx = [i for i in range(len(sizes)) if i != max_idx]
            # get index of all slices to exclude (keep max and up to two others if they are above threshold size)
            exclude_idx = [i for i in range(len(sizes)) if (i != max_idx) and (sizes[i] < threshold_size or sizes[i] < 10)]

            # find three biggest as the maximum number to be included per node
            if len(patient_node_slice_sizes[node]) - len(exclude_idx) > 3:
                sizes_with_indices = [(item[0], idx) for idx, item in enumerate(patient_node_slice_sizes[node])]
                sizes_with_indices.sort(reverse=True, key=lambda x: x[0])
                top_3_indices = [item[1] for item in sizes_with_indices[:3]]
                exclude_idx = [i for i in exclude_idx if i not in top_3_indices]

            if len(exclude_idx) > 3:
                print(sizes)
                print('exclude', [sizes[idx] for idx in exclude_idx])
                print('original node', [item[0] for item in patient_node_slice_sizes['_'.join(''.join(['\mask//', node.split('//')[1]]).split('_')[:-1])]])


            #print(max_size, [sizes[i] for i in range(len(sizes)) if i != max_idx and sizes[i] < threshold_size], [sizes[i] for i in range(len(sizes)) if i != max_idx and sizes[i] >= threshold_size])
            # print('max', patient_node_slice_sizes[node][max_idx])
            # print('nonmax', [patient_node_slice_sizes[node][idx] for idx in non_max_idx])
            files_to_skip.extend([patient_node_slice_sizes[node][idx][1] for idx in exclude_idx])
            non_max_total.extend([patient_node_slice_sizes[node][idx][1] for idx in non_max_idx])


    print('Skipped files in MLP data as other slices are much bigger for same node:', len(files_to_skip))
    print('Non max files:', len(non_max_total))




    # get ids of patients
    patient_files_list = [f for f in os.listdir(IMAGE_DIR + '\mri')] + [f  for f in os.listdir(IMAGE_DIR + '\mri_aug')]
    patient_ids = []
    for i in range(len(patient_files_list)):
        if patient_files_list[i][:10] not in patient_ids:
            patient_ids.append(patient_files_list[i][:10])

    # MLP data - augment to get synthetic patients
    patient_slices_dict, patient_labels_dict, patient_file_names_dict, mask_sizes_patients, labels = augment_data(cohort1, all_files_list, files_to_skip, patient_ids, mask_sizes, num_synthetic=num_synthetic, oversample_ratio=oversample_ratio)

    # calculate the short and long axis diameters and the ratio of them for each node
    short_long_axes_dict = {}
    # load bounding box data
    bboxes = np.load(r"C:\Users\mm17b2k.DS\Documents\ARCANE_Data\bounding_boxes2.npy", allow_pickle=True).item()

    # pre-calculated min and max values for short and long axis diameters and the ratio of them
    min_S, max_S = np.round(1.6206887424795668, 3), np.round(14.787397134046275, 3)
    min_L, max_L = np.round(3.6764972460210004, 3), np.round(19.82188472370879, 3)
    # min_R, max_R = np.round(1.2330743154590893, 3), np.round(2.5113486556814064, 3)
    min_R, max_R = 0, np.round(0.8109811286010666, 3)


    # max_short, max_long, max_ratio = 0, 0, 0
    # min_short, min_long, min_ratio = 100, 100, 100
    single_file_names = list(set(['_'.join(file.split('//')[1].split('_')[0:4]) for file in all_files_list]))
    for patient_file in bboxes.keys():
        check_file_name = '_'.join(patient_file.split('\\')[-1].split('.')[0].split('_')[:-1])
        if check_file_name not in single_file_names:
            continue
        patient = patient_file.split('\\')[-1].split('_')[0]
        patient_short_long_axes_dict = {}
        for node in bboxes[patient_file].keys():
            # voxel size is 3.3mm x 0.573mm x 0.573mm
            sizes = np.array(bboxes[patient_file][node]['size'])*np.array([3.3, 0.573, 0.573])
            a, b, c = sizes
            long = np.sqrt(a**2 + b**2 + c**2) # long axis diameter with 3D Pythagoras
            pairs = [(a, b), (a, c), (b, c)]
            short = min([np.sqrt(pair[0]**2 + pair[1]**2) for pair in pairs]) # short axis diameter
            ratio = short/long

            # if short > max_short:
            #     max_short = short
            # if long > max_long:
            #     max_long = long
            # if ratio > max_ratio:
            #     max_ratio = ratio
            # if short < min_short:
            #     min_short = short
            # if long < min_long:
            #     min_long = long
            # if ratio < min_ratio:
            #     min_ratio = ratio
            # print([(long-min_L)/(max_L-min_L), (short-min_S)/(max_S-min_S), (ratio-min_R)/(max_R-min_R)])
            patient_short_long_axes_dict[node] = [(long-min_L)/(max_L-min_L), (short-min_S)/(max_S-min_S), (ratio-min_R)/(max_R-min_R)]
        short_long_axes_dict[patient] = patient_short_long_axes_dict
    # print('Max short:', max_short, 'Max long:', max_long, 'Max ratio:', max_ratio)
    # print('Min short:', min_short, 'Min long:', min_long, 'Min ratio:', min_ratio)

    pure_labels = []
    for i in range(len(patient_ids)):
        N = cohort1[cohort1[('shortpatpseudoid')] == patient_ids[i]]['NodeLabel'].item()
        if N == '0':
            pure_labels.append(0)
        else:
            pure_labels.append(1)

    if first_time_train_test_split:
        # train-test split with balanced classes
        train_ids, test_ids, train_labels, test_labels = train_test_split(patient_ids, pure_labels, test_size=0.35, stratify=pure_labels, random_state=1)
    if not cross_val and not first_time_train_test_split:
        #get test labels
        test_labels = []
        for i in range(len(patient_ids)):
            if patient_files_list[i][:10] in test_ids:
                N = cohort1[cohort1[('shortpatpseudoid')] == patient_ids[i]]['NodeLabel'].item()
                if N == '0':
                    test_labels.append(0)
                else:
                    test_labels.append(1)
    if cross_val:
        train_ids, test_ids, train_labels, test_labels = fold_data


    train_test_split_dict = {'train': train_ids, 'test': test_ids}

    # add mask_sizes dict with all the mask sizes for a given patient id
    mask_sizes_dict = {}
    total_zeros_orig, total_zeros_aug = 0, 0
    for key in mask_sizes.keys():
        if 'aug' in key:
            patient_id = '_'.join([key.split('//')[1].split('_')[0], ''.join(['aug', key.split('_')[-1][:-4]])])
            if mask_sizes[key] == 0:
                total_zeros_aug += 1
                print('augmented mask size is 0:', mask_sizes[key], key)
            #continue
        else:
            patient_id = key.split('//')[1].split('_')[0]
            if mask_sizes[key] == 0:
                total_zeros_orig += 1

        if patient_id in mask_sizes_dict.keys():
            mask_sizes_dict[patient_id].append(mask_sizes[key])
        else:
            mask_sizes_dict[patient_id] = [mask_sizes[key]]
    assert total_zeros_aug == total_zeros_orig == 0



    get_class_distribution(train_ids, 'Train', patient_labels_dict)
    mlp_train_ids = [key for key in patient_labels_dict.keys() if key[:10] in train_ids]
    mlp_train_labels = [patient_labels_dict[key] for key in train_ids]
    total_mlp_train_labels = [patient_labels_dict[key] for key in mlp_train_ids]
    print('MLP train scans + N synthetic patients:', len(mlp_train_ids), 'Positive samples:', sum(total_mlp_train_labels), 'Negative samples:', len(total_mlp_train_labels) - sum(total_mlp_train_labels), 'Positive proportion:', sum(total_mlp_train_labels)/len(total_mlp_train_labels))
    get_class_distribution(test_ids, 'Test', patient_labels_dict)

    # VAE data
    # redefine all files so it can have repeated scans for each patient to maximise data
    all_files_list = ['\mri' + '//' + f for f in os.listdir(IMAGE_DIR + '\mri')] + ['\mri_aug' + '//' + f  for f in os.listdir(IMAGE_DIR + '\mri_aug')]
    all_files_list.sort()
    train_images, test_images = [], []
    train_samples, train_aug_samples, test_samples = 0, 0, 0
    for file in all_files_list:
        if file.split('//')[1][:10] in train_ids:
            if 'mri_aug' in file:
                train_aug_samples += 1
            else:
                train_samples += 1
            train_images.append(file)
        if file.split('//')[1][:10] in test_ids:
            if 'mri_aug' not in file: # don't use augmented images for testing
                test_images.append(file)
                test_samples += 1
    print('VAE 2D patch lymph node data: Train samples:', train_samples, 'Train augmented samples:', train_aug_samples, 'Test samples:', test_samples)

    return patient_slices_dict, patient_labels_dict, patient_file_names_dict, short_long_axes_dict, mlp_train_ids, test_ids, mlp_train_labels, test_labels, train_images, test_images, train_test_split_dict, mask_sizes_patients

class Load_CNN_Images():
    def __init__(self, patient_file_names_dict, patient_labels, patient_ids, cohort1, all_files_list, short_long_axes_dict, mask_sizes, clinical_data_options, max_nodes=25):
        self.patient_ids = patient_ids
        self.patient_file_names_dict = patient_file_names_dict
        self.patient_labels = patient_labels
        self.cohort1 = cohort1
        self.all_files_list = all_files_list
        self.short_long_axes_dict = short_long_axes_dict
        self.mask_sizes = mask_sizes
        self.max_nodes = max_nodes
        self.clinical_data_options = clinical_data_options

        with open(r"C:\Users\mm17b2k.DS\Documents\ARCANE_Data\border_metrics.pkl", 'rb') as f:
            compactness, convexity = pickle.load(f)

        self.compactness = compactness
        self.convexity = convexity

        self.IMAGE_DIR = r"C:\Users\mm17b2k.DS\Documents\ARCANE_Data\Cohort1_2D_slices"

    def __len__(self):
        return len(self.patient_ids)

    def load_and_preprocess_nii(self, file_path):
        # Load the .nii file
        nii_img = nib.load(file_path)
        img_data = nii_img.get_fdata()
        img_data = torch.tensor(img_data, dtype=torch.float32)

        return img_data

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        file_names = self.patient_file_names_dict[patient_id]
        if len(file_names) > self.max_nodes:
            mask_sizes = self.mask_sizes[patient_id]
            sizes = sorted(enumerate(mask_sizes), key=lambda x: x[1], reverse=True)
            biggest_n_mask_idx = [i for i, size in sizes[:self.max_nodes]]
            file_names = sorted([file_names[i] for i in biggest_n_mask_idx])

        # add clinical data to the latent vectors
        patient_indicator = False
        patient_options = []
        if "patient" in self.clinical_data_options:
            patient_indicator = True
            patient_options.append('age_scaled')
            patient_options.append('sex_numeric')
        if "T_stage" in self.clinical_data_options:
            patient_indicator = True
            patient_options.append('TumourLabel_numeric')

        if patient_options == []:
            patient_clinical_data = None
        else:
            patient_clinical_data = self.cohort1[self.cohort1['shortpatpseudoid'] == patient_id.split('_')[0]][patient_options].values.tolist()

        LN_features_original = []
        for file_name in file_names:
            pat_id = patient_id.split('_')[0]
            node_number = float(file_name.split('//')[1].split('_')[6])
            long, short, ratio = self.short_long_axes_dict[pat_id][node_number]
            mask_file = file_name.replace('mri', 'mask')
            compactness = self.compactness[mask_file]
            convexity = self.convexity[mask_file]

            node_indicator = False
            node_options = []
            if "size" in self.clinical_data_options:
                node_indicator = True
                node_options.append(long)
                node_options.append(short)
                node_options.append(ratio)
            if "border" in self.clinical_data_options:
                node_indicator = True
                node_options.append(compactness)
                node_options.append(convexity)

            LN_features_original.append(node_options)

            #clinical_features.append([long, short, ratio, compactness, convexity]) #, patient_clinical_data[0][0], patient_clinical_data[0][1], patient_clinical_data[0][2]])


        LN_features = np.array(LN_features_original)
        if patient_indicator == True and ("size" in self.clinical_data_options):
            patient_clinical_data = patient_clinical_data[0] + LN_features[np.argmax(LN_features[:, 0])].tolist() # add the data for node with max long axis ratio (and corresponding short/ratio/compactness/convexity) to patient level clinical data
        if patient_indicator == True and ("border" in self.clinical_data_options) and ("size" not in self.clinical_data_options):
            patient_clinical_data = patient_clinical_data[0] + LN_features[np.argmin(LN_features[:, 0])].tolist() # add min compactness node data to patient level clinical data
        if patient_indicator == True and node_indicator == False:
            patient_clinical_data = patient_clinical_data[0]
        if patient_indicator == False and node_indicator == True:
            if "size" in self.clinical_data_options:
                patient_clinical_data = LN_features[np.argmax(LN_features[:, 0])].tolist()
            if ("border" in self.clinical_data_options) and ("size" not in self.clinical_data_options):
                patient_clinical_data = LN_features[np.argmin(LN_features[:, 0])].tolist()
        if patient_indicator == False and node_indicator == False:
            patient_clinical_data = []


        images = []
        for file_name in file_names:
            img = self.load_and_preprocess_nii(self.IMAGE_DIR + '\\' + file_name)
            img = img.permute(2, 0, 1)
            images.append(img)

        features = torch.stack(images)
        label = self.patient_labels[patient_id]

        # if ("size" in self.clinical_data_options) or ("border" in self.clinical_data_options):
        #     features = np.concatenate((features, LN_features), axis=1)

        number_of_nodes = len(features)
        if len(features) < self.max_nodes:
            # Calculate how many zeros to pad
            padding_size = self.max_nodes - number_of_nodes

            # Apply padding (only on the batch dimension)
            features = F.pad(features, (0, 0, 0, 0, 0, 0, 0, padding_size))
            if LN_features_original != []:
                for i in range(padding_size):
                    LN_features_original.append([0]*len(LN_features_original[0]))

        return features, torch.tensor(label, dtype=torch.long), torch.tensor(patient_clinical_data, dtype=torch.float32), torch.tensor(number_of_nodes, dtype=torch.float32), torch.tensor(LN_features_original, dtype=torch.float32)