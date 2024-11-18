import torch
import torch.nn as nn
from torch import optim
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import matplotlib.pyplot as plt
import numpy as np
import math
import os
import nibabel as nib
import pandas as pd
import time

from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve
from scipy.spatial.distance import cdist

from utils.loss_functions import loss_function
from utils.utility_code import weights_init, plot_results,  get_single_scan_file_list
from utils.datasets import LoadImages, Load_Latent_Vectors

import wandb


def train(epoch, optimiser, train_loader, vae_model, hyperparams, device, epochs, results_path, Run, MLP_training=False, mlp_model=None, mlp_hyperparams=None, mlp_data_dictionaries=None):
    '''
    Function to run one epoch of training on given data

    Input:
    - epoch: the epoch id
    - optimiser: optimiser used (default Adam)
    - train_loader: DataLoader for training data
    - vae_model: VAE model used
    - hyperparams: dictionary of hyperparameters of model

    Output:
    - Current training loss
    - SSIM score
    - KL divergence
    '''
    vae_model.train()     # Put model in training mode
    train_loss_list, recon_loss_list, kld_list, ssim_list = [], [], [], []
    beta_train_loss_list, alpha_recon_list, beta_kld_list, alpha_SSIM_list = [], [], [], []
    latent_size = hyperparams['latent_size']
    base = hyperparams['base']
    accumulation_steps = hyperparams['accumulation_steps']
    batch_size = hyperparams['batch_size']
    steps = 0

    # if MLP_training:
    #     latent_vectors = np.empty((len(train_loader.dataset), latent_size*base))
    #     batch_size = hyperparams['batch_size']

    # Iterate through training DataLoader and apply model and loss function
    for batch_idx, data in enumerate(train_loader):
        steps += 1
        data = data.float().to(device)
        optimiser.zero_grad()
        recon_batch, mu, logvar = vae_model(data)

        # if MLP_training:
        #     # save latent vectors
        #     latent_vectors[batch_idx*batch_size:(batch_idx+1)*batch_size] = torch.squeeze(torch.squeeze(mu, dim=2), dim=2).cpu().numpy()

        #print(recon_batch)
        # Calculate loss and save various metrics
        pure_loss, beta_vae_loss, recon_loss, alpha_recon, kld, beta_annealed_kld, ssim_score, alpha_ssim = loss_function(recon_batch, data, mu, logvar, epoch, epochs, hyperparams)

        beta_vae_loss.backward()  # Backpropagate loss

        # print gradients
        # for name, param in vae_model.named_parameters():
        #     if param.grad is not None:
        #         gradient_norm = torch.norm(param.grad)
        #         print(f"Magnitude of gradient of {name}: {gradient_norm.item()}")
        train_loss_list.append(pure_loss.item())
        beta_train_loss_list.append(beta_vae_loss.item())
        recon_loss_list.append(recon_loss.item())
        alpha_recon_list.append(alpha_recon.item())
        kld_list.append(kld.item())
        beta_kld_list.append(beta_annealed_kld.item())
        ssim_list.append(ssim_score.item())
        alpha_SSIM_list.append(alpha_ssim.item())

        if steps % accumulation_steps == 1:
            optimiser.step() # Update optimisation function
            optimiser.zero_grad()

        #optimiser.step() # Update optimisation function
        # if batch_idx % 50 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tPure Loss: {:.6f}, Beta Loss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader),
        #         pure_loss.item(), loss.item()))
        if math.isnan(pure_loss):
            break

    optimiser.step()
    optimiser.zero_grad()

    # Sample and display input vs reconstruction, error plot and synthetic example
    if not os.path.exists(results_path + "/train_images/" + str(Run)):
        os.makedirs(results_path + "/train_images/" + str(Run))
    if((epoch%50==1) or (epoch==25) or (epoch < 1) or (epoch==epochs-1)) or MLP_training:
        print('12 Real Images')
        img_grid = make_grid(data[:12], nrow=4, padding=12, pad_value=-1)
        plt.figure(figsize=(10,5))
        plt.imshow(img_grid[0].detach().cpu(), cmap='gray')
        plt.axis('off')
        plt.savefig(results_path + "/train_images/" + str(Run) + "/visualise_real_" + str(epoch) + '.png')
        plt.show()


        print('12 Reconstructed Images')
        img_grid = make_grid(recon_batch[:12], nrow=4, padding=12, pad_value=-1)
        plt.figure(figsize=(10,5))
        plt.imshow(img_grid[0].detach().cpu(), cmap='gray')
        plt.axis('off')
        plt.savefig(results_path + "/train_images/" + str(Run) + "/visualise_recon_" + str(epoch) + '.png')
        plt.show()


        print('12 Synthetic Images')
        sample = torch.randn((12, latent_size*base, 1, 1)).to(device)
        sample.to(device)
        recon_rand_sample = vae_model.decode(sample)
        img_grid = make_grid(recon_rand_sample[:12], nrow=4, padding=12, pad_value=-1)
        fig = plt.figure(figsize=(10,5))
        plt.imshow(img_grid[0].detach().cpu(), cmap='gray')
        plt.axis('off')
        plt.savefig(results_path + "/train_images/" + str(Run) + "/visualise_synth_" + str(epoch) + '.png') #, dpi=100)
        plt.show()

    # Give epoch summary of metrics using running losses
    train_loss = np.sum(train_loss_list)/len(train_loader.dataset)
    beta_train_loss = np.sum(beta_train_loss_list)/len(train_loader.dataset)
    recon_loss = np.sum(recon_loss_list)/len(train_loader.dataset)
    alpha_recon = np.sum(alpha_recon_list)/len(train_loader.dataset)
    kld = np.sum(kld_list)/len(train_loader.dataset)
    beta_annealed_kld = np.sum(beta_kld_list)/len(train_loader.dataset)
    alpha_SSIM = np.sum(alpha_SSIM_list)/len(train_loader.dataset)
    ssim_mean = np.mean(ssim_list)
    #alpha_SSIM = np.mean(alpha_SSIM_list)


    print('====> Epoch {}'.format(epoch))
    # print('====> Pure Train Loss: {:.4f}'.format(train_loss), 'Beta Train Loss: {:.4f}'.format(beta_train_loss), 'Train SSIM: {:.6f}'.format(ssim_mean))
    # print('====> Recon Loss: {:.4f}'.format(recon_loss), 'KLD: {:.4f}'.format(kld))
    print("Train Loss: KLD + Recon =: {:.6f} + {:.6f} = {:.4f}".format(kld, recon_loss, train_loss),
          "Beta Train Loss: KLD + Recon + SSIM =: {:.6f} + {:.6f} + {:.6f} = {:.4f}".format(beta_annealed_kld, alpha_recon, alpha_SSIM,  beta_train_loss),
          "SSIM Score: {:.6f}".format(ssim_score.item())) #, "MS-SSIM Score: {:.6f}".format(ms_ssim_score.item()))


    if MLP_training:
        IMAGE_DIR = r"C:\Users\mm17b2k.DS\Documents\ARCANE_Data\Cohort1_2D_slices"
        cohort1 = pd.read_excel(r"C:\Users\mm17b2k.DS\Documents\ARCANE_Data\Cohort1.xlsx")
        all_files_list = ['\mri' + '//' + f for f in os.listdir(IMAGE_DIR + '\mri')] + ['\mri_aug' + '//' + f  for f in os.listdir(IMAGE_DIR + '\mri_aug')]
        all_files_list.sort()
        all_files_list = get_single_scan_file_list(all_files_list, IMAGE_DIR, cohort1)
        #VAE_metrics, latent_vectors = evaluate_VAE(vae_model, train_loss, results_path, batch_size, device, IMAGE_DIR, all_files_list, feature_length=latent_size*base, Run=Run)

        max_node_slices = mlp_hyperparams['max_node_slices']
        accumulation_steps = mlp_hyperparams['accumulation_steps']
        batch_size = mlp_hyperparams['batch_size']
        max_node_slices = mlp_hyperparams['max_node_slices']
        clinical_data_options = mlp_hyperparams['clinical_data_options']
        threshold = mlp_hyperparams['threshold']

        patient_slices_dict = mlp_data_dictionaries["slices"]
        patient_labels_dict = mlp_data_dictionaries["labels"]
        patient_file_names_dict = mlp_data_dictionaries["files"]
        short_long_axes_dict = mlp_data_dictionaries["sizes"]
        mask_sizes = mlp_data_dictionaries["mask_sizes"]
        clinical_data_options = mlp_data_dictionaries["clinical_data_options"]
        mlp_train_ids = mlp_data_dictionaries["mlp_train_ids"]
        test_ids = mlp_data_dictionaries["test_ids"]

        latent_vectors = None
        train_dataset = Load_Latent_Vectors(patient_slices_dict, latent_vectors, patient_labels_dict, mlp_train_ids, cohort1, all_files_list, short_long_axes_dict, mask_sizes, clinical_data_options, max_nodes=max_node_slices, add_images=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = Load_Latent_Vectors(patient_slices_dict, latent_vectors, patient_labels_dict, test_ids, cohort1, all_files_list, short_long_axes_dict, mask_sizes, clinical_data_options, max_nodes=max_node_slices, add_images=True)
        test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

        mlp_criterion = nn.BCELoss()
        mlp_optimiser = torch.optim.Adam(mlp_model.parameters(), lr=mlp_hyperparams["lr"], weight_decay=mlp_hyperparams["weight_decay"])


        steps = 0
        mlp_model.train()
        all_train_labels, all_train_preds, all_train_probs = [], [], []
        mlp_train_loss = 0
        optimiser.zero_grad()
        mlp_optimiser.zero_grad()

        train_loss_list, recon_loss_list, kld_list, ssim_list = [], [], [], []
        beta_train_loss_list, alpha_recon_list, beta_kld_list, alpha_SSIM_list = [], [], [], []

        for LN_features, label, clinical_data, number_of_nodes, images in train_dataloader:
            images = images.float().to(device)
            label = label.to(device)
            clinical_data, number_of_nodes, LN_features = clinical_data.to(device), number_of_nodes.to(device), LN_features.to(device)
            current_batch_size = images.shape[0]
            images = images.reshape(-1, 1, 32, 32)

            recon_batch, features, logvar = vae_model(images)
            pure_loss, beta_vae_loss, recon_loss, alpha_recon, kld, beta_annealed_kld, ssim_score, alpha_ssim = loss_function(recon_batch, images, features, logvar, epoch, epochs, hyperparams)

            train_loss_list.append(pure_loss.item())
            beta_train_loss_list.append(beta_vae_loss.item())
            recon_loss_list.append(recon_loss.item())
            alpha_recon_list.append(alpha_recon.item())
            kld_list.append(kld.item())
            beta_kld_list.append(beta_annealed_kld.item())
            ssim_list.append(ssim_score.item())
            alpha_SSIM_list.append(alpha_ssim.item())

            steps += 1
            # Forward pass
            if ("size" in clinical_data_options) or ("border" in clinical_data_options):
                features = features.view(current_batch_size, max_node_slices, latent_size * base)
                features = torch.cat((features, LN_features), dim=2)


            output, max_vals, attentions, classifications = mlp_model(features, clinical_data, number_of_nodes, label)
            output = output.squeeze(1)

            mlp_loss = mlp_criterion(output, label.float()) #*weight
            mlp_train_loss += mlp_loss.item()

            beta_vae_loss = beta_vae_loss + 0.1*mlp_loss # downweight MLP loss

            beta_vae_loss.backward()  # Backpropagate loss
            #beta_vae_loss.backward(retain_graph=True)
            # mlp_loss.backward()

            #if steps % accumulation_steps == 1:
            if steps > 0:
                optimiser.step()
                optimiser.zero_grad()
                mlp_optimiser.step()
                mlp_optimiser.zero_grad()


            predicted_probs = output
            classifications_class = (classifications >= threshold).long()
            #predicted_probs = 0.6*max_vals + 0.35*classifications_class + 0.05*attentions
            predicted_class = (predicted_probs >= threshold).long()


            # Store predictions and labels
            all_train_labels.extend(label.cpu().numpy())
            all_train_preds.extend(predicted_class.cpu().numpy())
            all_train_probs.extend(predicted_probs.tolist())

        # Give epoch summary of metrics using running losses
        train_loss = np.sum(train_loss_list)/len(train_loader.dataset)
        beta_train_loss = np.sum(beta_train_loss_list)/len(train_loader.dataset)
        recon_loss = np.sum(recon_loss_list)/len(train_loader.dataset)
        alpha_recon = np.sum(alpha_recon_list)/len(train_loader.dataset)
        kld = np.sum(kld_list)/len(train_loader.dataset)
        beta_annealed_kld = np.sum(beta_kld_list)/len(train_loader.dataset)
        alpha_SSIM = np.sum(alpha_SSIM_list)/len(train_loader.dataset)
        ssim_mean = np.mean(ssim_list)
        #alpha_SSIM = np.mean(alpha_SSIM_list)


        print('====> Epoch {}'.format(epoch))
        # print('====> Pure Train Loss: {:.4f}'.format(train_loss), 'Beta Train Loss: {:.4f}'.format(beta_train_loss), 'Train SSIM: {:.6f}'.format(ssim_mean))
        # print('====> Recon Loss: {:.4f}'.format(recon_loss), 'KLD: {:.4f}'.format(kld))
        print("Train Loss: KLD + Recon =: {:.6f} + {:.6f} = {:.4f}".format(kld, recon_loss, train_loss),
              "Beta Train Loss: KLD + Recon + SSIM =: {:.6f} + {:.6f} + {:.6f} = {:.4f}".format(beta_annealed_kld, alpha_recon, alpha_SSIM,  beta_train_loss),
              "SSIM Score: {:.6f}".format(ssim_score.item())) #, "MS-SSIM Score: {:.6f}".format(ms_ssim_score.item()))



        mlp_optimiser.step()
        mlp_optimiser.zero_grad()
        optimiser.step()
        optimiser.zero_grad()
        print('MLP learning rate:', mlp_optimiser.param_groups[0]['lr'])
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_auc = roc_auc_score(all_train_labels, all_train_preds)
        train_bal_accuracy = balanced_accuracy_score(all_train_labels, all_train_preds)
        train_confusion_matrix = confusion_matrix(all_train_labels, all_train_preds)
        tn, fp, fn, tp = confusion_matrix(all_train_labels, all_train_preds).ravel()
        # Compute sensitivity (recall) and specificity
        train_sensitivity = tp / (tp + fn)
        train_specificity = tn / (tn + fp)


        print(f'Epoch: {epoch}, Train: Loss: {train_loss/len(train_dataloader):.4f}, Accuracy: {train_accuracy:.4f}, Balanced Accuracy: {train_bal_accuracy:.4f}, AUC: {train_auc:.4f}, Sensitivity: {train_sensitivity:.4f}, Specificity: {train_specificity:.4f}')
        print(f'Train Confusion Matrix:')
        print(train_confusion_matrix)


        # Evaluation phase
        mlp_model.eval()
        mlp_test_loss = 0
        all_test_labels = []
        all_test_preds = []
        all_test_probs = []
        with torch.no_grad():
            for LN_features, label, clinical_data, number_of_nodes, images in test_dataloader:
                images = images.float().to(device)
                label = label.to(device)
                clinical_data, number_of_nodes, LN_features = clinical_data.to(device), number_of_nodes.to(device), LN_features.to(device)
                current_batch_size = images.shape[0]
                images = images.reshape(-1, 1, 32, 32)

                recon_batch, features, logvar = vae_model(images)

                #features = features.to(device)
                if ("size" in clinical_data_options) or ("border" in clinical_data_options):
                    features = features.view(current_batch_size, max_node_slices, latent_size * base)
                    features = torch.cat((features, LN_features), dim=2)

                output, max_vals, attentions, classifications = mlp_model(features, clinical_data, number_of_nodes, label)
                output = output.squeeze(1)

                #output = output.squeeze(0)
                loss = mlp_criterion(output, label.float())
                mlp_test_loss += loss.item()


                # Store predictions and labels
                #predicted_probs = F.softmax(output, dim=1)[:, 1]  # Probability of class 1 (positive)
                #predicted_probs = torch.sigmoid(output)
                predicted_probs = output
                classifications_class = (classifications >= threshold).long()
                #predicted_probs = 0.6*max_vals + 0.35*classifications_class + 0.05*attentions
                predicted_class = (predicted_probs >= threshold).type(torch.long)
                all_test_labels.extend(label.cpu().numpy())
                all_test_preds.extend(predicted_class.cpu().numpy())
                all_test_probs.extend(predicted_probs.cpu().numpy())

                # random_int = np.random.randint(1, 8)
                # if random_int == 1:
                #     rdn_idx  = np.random.randint(0, len(features))
                #     print(rdn_idx, 'label (test)', label[rdn_idx].item(), 'output', output[rdn_idx].item(), 'predicted class', predicted_class[rdn_idx].item(), 'max', max_vals[rdn_idx].item(), 'attention', attentions[rdn_idx].item(), 'classification', classifications[rdn_idx].item(), 'class binary', classifications_class[rdn_idx].item(), 'number of nodes', number_of_nodes[rdn_idx].item()) #'reweighted prediction', predicted_probs[rdn_idx].item())
                #

        test_accuracy = accuracy_score(all_test_labels, all_test_preds)
        test_auc = roc_auc_score(all_test_labels, all_test_preds)
        test_bal_accuracy = balanced_accuracy_score(all_test_labels, all_test_preds)
        test_confusion_matrix = confusion_matrix(all_test_labels, all_test_preds)
        tn, fp, fn, tp = confusion_matrix(all_test_labels, all_test_preds).ravel()
        # Compute sensitivity (recall) and specificity
        test_sensitivity = tp / (tp + fn)
        test_specificity = tn / (tn + fp)
        #if epoch % 5 == 0 or epoch + 20 > num_epochs-1:
        print(f'Test: Loss: {mlp_test_loss/len(test_dataloader):.4f}, Accuracy: {test_accuracy:.4f}, Balanced Accuracy: {test_bal_accuracy:.4f}, AUC: {test_auc:.4f}, Sensitivity: {test_sensitivity:.4f}, Specificity: {test_specificity:.4f}')
        print('Test Confusion Matrix:')
        print(test_confusion_matrix)
        # Wait for GPU to cool down for 10 seconds
        time.sleep(10)

    return beta_train_loss, train_loss, ssim_mean, kld, recon_loss, vae_model


def test(epoch, test_loader, vae_model, hyperparams, device, epochs):
    '''
    Function to test model after each epoch

    Input:
    - epoch: the epoch id
    - test_loader: DataLoader for test data
    - vae_model: VAE model used
    - hyperparams: dictionary of hyperparameters of model

    Output:
    - Test loss
    - SSIM score
    '''
    vae_model.eval()     # Put model in testing mode
    test_loss_list, recon_loss_list, kld_list, ssim_list = [], [], [], []
    beta_test_loss_list, alpha_recon_list, beta_kld_list, alpha_SSIM_list = [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.float().to(device)
            recon_batch, mu, logvar = vae_model(data)
            pure_loss, beta_vae_loss, recon_loss, alpha_recon, kld, beta_annealed_kld, ssim_score, alpha_ssim = loss_function(recon_batch, data, mu, logvar, epoch, epochs, hyperparams)
            test_loss_list.append(pure_loss.item())
            beta_test_loss_list.append(beta_vae_loss.item())
            recon_loss_list.append(recon_loss.item())
            alpha_recon_list.append(alpha_recon.item())
            kld_list.append(kld.item())
            beta_kld_list.append(beta_annealed_kld.item())
            ssim_list.append(ssim_score.item())
            alpha_SSIM_list.append(alpha_ssim.item())
            if math.isnan(pure_loss):
                break

    # Give epoch summary of metrics using running losses
    test_loss = np.sum(test_loss_list)/len(test_loader.dataset)
    beta_test_loss = np.sum(beta_test_loss_list)/len(test_loader.dataset)
    recon_loss = np.sum(recon_loss_list)/len(test_loader.dataset)
    alpha_recon = np.sum(alpha_recon_list)/len(test_loader.dataset)
    kld = np.sum(kld_list)/len(test_loader.dataset)
    beta_annealed_kld = np.sum(beta_kld_list)/len(test_loader.dataset)
    alpha_SSIM = np.sum(alpha_SSIM_list)/len(test_loader.dataset)
    ssim_mean = np.mean(ssim_list)
    # print('====> Pure Test Loss: {:.4f}'.format(test_loss), 'Beta Test Loss: {:.4f}'.format(beta_test_loss), 'Test SSIM: {:.6f}'.format(ssim_mean))
    # print('====> Recon Loss: {:.4f}'.format(recon_loss), 'KLD: {:.4f}'.format(kld))
    print("Test Loss: KLD + Recon =: {:.6f} + {:.6f} = {:.4f}".format(kld, recon_loss, test_loss),
          "Beta Test Loss: KLD + Recon + SSIM =: {:.6f} + {:.6f} + {:.6f} = {:.4f}".format(beta_annealed_kld, alpha_recon, alpha_SSIM,  beta_test_loss),
          "SSIM Score: {:.6f}".format(ssim_score.item()))

    return test_loss, ssim_mean, kld, recon_loss, beta_test_loss



def train_VAE_model(vae_model, epochs, train_loader, test_loader, hyperparams, device, results_path, save_results_path, sample_shape,  train_test_split_dict, wandb_sweep=False, Run=1, MLP_training=False, mlp_model=None, mlp_hyperparams=None, mlp_data_dictionaries=None, fold_idx=0):
    '''
    Function to run training procedure of multiple epochs
    Input:
    - model: VAE model
    - epochs: number of epochs to train over
    - Run: current run number in hyperparameter training regime
    - train_loader and test_loader: DataLoaders for train/test data
    - hyperparams: dictionary of hyperparameters of model

    Output:
    - test_loss: to check whether loss has exploaded (infinite/NaN) as will need to stop training
    - test_ssim: SSIM for quick summary of how well the model has done
    '''
    best_ssim = 0.71

    lr = hyperparams["lr"]
    weight_decay = hyperparams["weight_decay"]
    train_losses, test_losses, ssim_score_list = [], [], []
    optimiser = optim.Adam(vae_model.parameters(), lr=lr, weight_decay=weight_decay)
    # factor: how much to reduce learning rate, patients: how many epochs without improvement before reducing, threshold: for measuring the new optimum
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=30,
                                                           threshold=2e-6, threshold_mode='abs')

    counter = 0
    for epoch in range(1, epochs + 1):
        if MLP_training:
            beta_train_loss, train_loss, train_ssim, train_kld, train_recon, vae_model = train(epoch, optimiser, train_loader, vae_model, hyperparams, device, epochs, results_path, Run, MLP_training, mlp_model, mlp_hyperparams, mlp_data_dictionaries)
        else:
            beta_train_loss, train_loss, train_ssim, train_kld, train_recon, vae_model = train(epoch, optimiser, train_loader, vae_model, hyperparams, device, epochs, results_path, Run)
        test_loss, test_ssim, test_kld, test_recon, beta_test_loss = test(epoch, test_loader, vae_model, hyperparams, device, epochs)
        scheduler.step(train_recon)   # update learning rate scheduler based on training loss
        print('Learning Rate:', optimiser.param_groups[0]['lr'])
        VAE_metrics = {"Train Loss": train_loss, "Train SSIM": train_ssim, "Train KLD": train_kld, "Train Recon Loss": train_recon, "Train Beta Loss": beta_train_loss,
                       "Test Loss": test_loss, "Test SSIM": test_ssim, "Test KLD": test_kld, "Test Recon Loss": test_recon, "Test Beta Loss": beta_test_loss,
                       "lr": optimiser.param_groups[0]['lr']}

        if wandb_sweep:
            wandb.log(VAE_metrics)

        #counter = early_stopping(counter, train_losses[-1], train_loss, min_delta=0.0001)
        #if train_recon > 0.0001 or test_recon > 0.00015:
        if test_ssim < 0.68:
            counter += 1
            if counter % 10 == 0:
                print('Early Stopping Counter At:', counter)
            if counter > 25 and test_recon > 0.0005:
                print("Early Stopping at Epoch:", epoch)
                break
            if counter > 50 and test_recon > 0.00025:
                print("Early Stopping at Epoch:", epoch)
                break
            if counter > 75 and test_recon > 0.00015:
                print("Early Stopping at Epoch:", epoch)
                break
            if counter > 125 and test_recon > 0.0001:
                print("Early Stopping at Epoch:", epoch)
                break
            if counter > 150 and test_recon > 0.00007:
                print("Early Stopping at Epoch:", epoch)
                break
            if counter > 175 and test_recon > 0.00006:
                print("Early Stopping at Epoch:", epoch)
                break

        if math.isnan(train_loss):
            print('Training stopped due to infinite loss')
            break

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        ssim_score_list.append(test_ssim)

        if test_ssim > best_ssim + 0.01: # save model if SSIM improves by 0.005
            best_ssim = test_ssim
            save_results_path = rf"C:\Users\mm17b2k.DS\Documents\Python\ARCANE_Results\VAE_fold_{fold_idx}_run_{Run}.pt"

            save_results_path = save_results_path.replace(".pt", f"_ssim_{test_ssim}_epoch_{epoch}.pt")

            torch.save({"state_dict": vae_model.state_dict(), "train_losses": train_losses, "test_losses": test_losses,
                        "hyperparams": hyperparams, "train_test_split": train_test_split_dict}, save_results_path)

            plot_results(results_path, save_results_path, f'loss_graph_{Run}_epoch_{epoch}.jpg')

            IMAGE_DIR = r"C:\Users\mm17b2k.DS\Documents\ARCANE_Data\Cohort1_2D_slices"
            cohort1 = pd.read_excel(r"C:\Users\mm17b2k.DS\Documents\ARCANE_Data\Cohort1.xlsx")
            all_files_list = ['\mri' + '//' + f for f in os.listdir(IMAGE_DIR + '\mri')] + ['\mri_aug' + '//' + f  for f in os.listdir(IMAGE_DIR + '\mri_aug')]
            # only one scan per patient to avoid repeated scans when saving latent vectors
            all_files_list2 = get_single_scan_file_list(all_files_list, IMAGE_DIR, cohort1)
            all_files_list2.sort()
            batch_size = hyperparams['batch_size']
            latent_size = hyperparams['latent_size']
            base = hyperparams['base']
            feature_length = latent_size * base
            evaluate_VAE(vae_model, test_loss, results_path, batch_size, device, IMAGE_DIR, all_files_list, feature_length, Run, wandb_sweep=False, fold_idx=fold_idx, epoch=epoch)


    if test_ssim > 0.72:
        save_results_path = rf"C:\Users\mm17b2k.DS\Documents\Python\ARCANE_Results\VAE_fold_{fold_idx}_run_{Run}.pt"
        save_results_path = save_results_path.replace(".pt", f"_ssim_{test_ssim}.pt")

        torch.save({"state_dict": vae_model.state_dict(), "train_losses": train_losses, "test_losses": test_losses,
                "hyperparams": hyperparams, "train_test_split": train_test_split_dict}, save_results_path)

        plot_results(results_path, save_results_path, f'loss_graph_{Run}.jpg')

    return vae_model, test_loss, test_ssim, train_loss, train_ssim, VAE_metrics


def evaluate_VAE(vae_model, vae_test_loss, results_path, batch_size, device, IMAGE_DIR, all_files_list, feature_length, Run, wandb_sweep=False, fold_idx=-42, epoch=-42):
    images = LoadImages(main_dir=IMAGE_DIR + '/', files_list=all_files_list)
    image_loader = DataLoader(images, batch_size, shuffle=False)
    vae_model.eval()
    MSE = nn.MSELoss(reduction='mean')
    l1_loss = nn.L1Loss(reduction='mean')
    mus = np.empty((len(all_files_list), feature_length))
    SSIM_list, MSE_list, L1_list = [], [], []
    if not math.isnan(vae_test_loss):
        with torch.no_grad():
            for batch_idx, data in enumerate(image_loader):
                data = data.float().to(device)
                reconstructions_batch, mu_batch, log_var_batch = vae_model(data)
                # save latent vectors
                mus[batch_idx*batch_size:(batch_idx+1)*batch_size] = torch.squeeze(torch.squeeze(mu_batch, dim=2), dim=2).cpu().numpy()

                # calculate SSIM
                SSIM_batch = ssim(data, reconstructions_batch, data_range=1, nonnegative_ssim=True)
                SSIM_list.append(np.array(SSIM_batch.cpu()).item())
                # calculate MSE
                MSE_batch = MSE(data, reconstructions_batch)
                MSE_list.append(np.array(MSE_batch.cpu()).item())
                # calculate MAE
                L1_batch = l1_loss(data, reconstructions_batch)
                L1_list.append(np.array(L1_batch.cpu()).item())


        #print('Number of latent vectors', len(mus))
        print('Latent vector size', len(mus[0]))
        print('Mean Squared Error', np.mean(MSE_list))
        print('Mean Absolute Error', np.mean(L1_list))
        print('Mean SSIM', np.mean(SSIM_list))
        print('Number of latent vectors', len(mus))
        if wandb_sweep:
            wandb.log({"Mean SSIM": np.mean(SSIM_list), "MSE (Mean Squared Error)": np.mean(MSE_list), "MAE (Mean Absolute Error)": np.mean(L1_list)})

        if np.mean(SSIM_list) > 0.72:
            if epoch > 0:
                np.save(results_path + '/' + f'latent_vectors_fold_{fold_idx}_epoch_{epoch}_run_{Run}', mus)
            else:
                if fold_idx > -1:
                    np.save(results_path + '/' + f'latent_vectors_fold_{fold_idx}_run_{Run}', mus)
                else:
                    np.save(results_path + '/' + f'latent_vectors_run_{Run}', mus)

        metrics_list = [np.mean(SSIM_list), np.mean(MSE_list), np.mean(L1_list)]
        #np.save(results_path + '/' + 'VAEMetrics', metrics_list)
        return metrics_list, mus

def train_MLP_model(train_dataloader, test_dataloader, MLP_hyperparams, device, mlp_model):
    threshold = MLP_hyperparams['threshold']
    num_epochs = MLP_hyperparams['num_epochs']
    lr = MLP_hyperparams['lr']
    weight_decay = MLP_hyperparams['weight_decay']

    criterion = nn.BCELoss()
    optimiser = torch.optim.Adam(mlp_model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=10, verbose=True,
                                                              threshold=0.01, threshold_mode='abs')

    accumulation_steps = 3
    for epoch in range(1, num_epochs+1):
        # Training phase
        mlp_model.train()
        train_loss = 0
        all_train_labels = []
        all_train_preds = []
        steps = 0
        optimiser.zero_grad()

        for features, label in train_dataloader:
            steps += 1
            features, label = features.to(device), label.to(device)

            # Forward pass
            output = mlp_model(features.squeeze(0))  # Remove batch dimension

            if label == 1:
                weight = torch.tensor([10.0]).to(device)
            else:
                weight = torch.tensor([1.0]).to(device)

            loss = criterion(output, label.float())*weight
            train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            if steps % accumulation_steps == 1:
                optimiser.step()
                optimiser.zero_grad()

            # Apply threshold to determine predicted class
            predicted_probs = output #torch.sigmoid(output)
            predicted_class = (predicted_probs >= threshold).long()

            # Store predictions and labels
            all_train_labels.extend(label.cpu().numpy())
            all_train_preds.extend(predicted_class.cpu().numpy())


        optimiser.step()
        optimiser.zero_grad()
        lr_scheduler.step(train_loss/len(train_dataloader))
        print('Learning rate:', optimiser.param_groups[0]['lr'])
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_auc = roc_auc_score(all_train_labels, all_train_preds)
        train_bal_accuracy = balanced_accuracy_score(all_train_labels, all_train_preds)
        train_confusion_matrix = confusion_matrix(all_train_labels, all_train_preds)
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss/len(train_dataloader):.4f}, Train Accuracy: {train_accuracy:.4f}, Train Balanced Accuracy: {train_bal_accuracy:.4f}, Train AUC: {train_auc:.4f}')
        print(f'Train Confusion Matrix:')
        print(train_confusion_matrix)

        evaluate_MLP_model(test_dataloader, mlp_model, device, criterion, epoch, num_epochs, threshold=0.5)

    return mlp_model

def evaluate_MLP_model(test_dataloader, mlp_model, device, criterion, epoch, num_epochs, threshold=0.5):
    # Evaluation phase
    mlp_model.eval()
    test_loss = 0
    all_test_labels = []
    all_test_preds = []
    all_test_probs = []
    with torch.no_grad():
        for features, label in test_dataloader:
            features, label = features.to(device), label.to(device)
            output = mlp_model(features.squeeze(0))  # Remove batch dimension
            #output = output.squeeze(0)
            loss = criterion(output, label.float())
            test_loss += loss.item()

            # Store predictions and labels
            predicted_probs = output #torch.sigmoid(output)
            predicted_class = (predicted_probs >= threshold).type(torch.long)
            all_test_labels.extend(label.cpu().numpy())
            all_test_preds.extend(predicted_class.cpu().numpy())
            all_test_probs.extend(predicted_probs.cpu().numpy())

    test_accuracy = accuracy_score(all_test_labels, all_test_preds)
    test_auc = roc_auc_score(all_test_labels, all_test_preds)
    test_bal_accuracy = balanced_accuracy_score(all_test_labels, all_test_preds)
    test_confusion_matrix = confusion_matrix(all_test_labels, all_test_preds)
    print(f'Epoch [{epoch}/{num_epochs}], Test Loss: {test_loss/len(test_dataloader):.4f}, Test Accuracy: {test_accuracy:.4f}, Test Balanced Accuracy: {test_bal_accuracy:.4f}, Test AUC: {test_auc:.4f}')
    print('Test Confusion Matrix:')
    print(test_confusion_matrix)

    if epoch % 50 == 0 or epoch == num_epochs-1:
        prob_true, prob_pred = calibration_curve(all_test_labels, all_test_probs, n_bins=10)

        plt.figure(figsize=(10, 5))

        # Plot calibration curve
        plt.plot(prob_pred, prob_true, marker='o', label='MIL-MLP')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Predicted probability')
        plt.ylabel('True probability')
        plt.title('Calibration Curve')
        plt.legend()
        plt.show()

        # Plot distribution of predicted probabilities
        plt.hist(prob_pred, bins=5, range=(0, 1), edgecolor='k', alpha=0.7)
        plt.xlabel('Predicted probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Probabilities')

        plt.tight_layout()
        plt.show()
        plt.close()

def mixup_patient_data(data):
    alpha = 0.2 # parameter of beta distribution
    synth_patient = np.empty(data.shape)
    num_samples = data.shape[0]
    if num_samples == 1:
        synth_patient = data
    elif num_samples == 2:
        vec1 = data[0]
        vec2 = data[1]
        lamda = np.random.beta(alpha, alpha)
        mixed_vec1 = lamda * vec1 + (1 - lamda) * vec2
        lamda = np.random.beta(alpha, alpha)
        mixed_vec2 = lamda * vec1 + (1 - lamda) * vec2
        synth_patient[0] = mixed_vec1
        synth_patient[1] = mixed_vec2
    else:
        # calculate pairwise distances
        distances = cdist(data, data, metric='euclidean')
        closest_indices = np.argsort(distances, axis=1)[:, :]
        for j in range(num_samples):
            lamda = np.random.beta(alpha, alpha)
            if num_samples == 3:
                random_idx = np.random.randint(1, 3) # any of top two closest (not itself in index 0)
            else:
                random_idx = np.random.randint(1, 4) # any of top 3
            #print(num_samples,random_idx)
            vec1 = data[j]
            vec2 = data[closest_indices[j][random_idx]]
            mixed_vec = lamda * vec1 + (1 - lamda) * vec2
            synth_patient[j] = mixed_vec

    return torch.tensor(synth_patient)
def mixup_batch(features):
    batch_size, num_nodes, feature_size = features.shape
    processed_tensor = []

    for i in range(batch_size):
        patient_data = np.array(features[i])

        random_int = np.random.randint(0, 2) # ~50% chance of mixup
        if random_int == 0:
            # Skip mixup for this patient
            processed_tensor.append(features[i])
            continue

        # Filter out rows that are all zeros
        non_zero_rows = np.any(patient_data != 0, axis=1)
        filtered_data = patient_data[non_zero_rows]

        # Apply mixup_patient_data function
        mixed_data = mixup_patient_data(filtered_data)

        # Create a padded tensor of shape (num_nodes, feature_size)
        padded_mixed_data = torch.zeros((num_nodes, feature_size))
        padded_mixed_data[:mixed_data.shape[0], :] = mixed_data

        # Append to the processed tensor list
        processed_tensor.append(padded_mixed_data)

    # Stack all processed patients back into a tensor of shape (batch_size, num_nodes, feature_size)
    processed_tensor = torch.stack(processed_tensor)

    return processed_tensor



def process_batch_with_noise(features, mean=0.0, std=0.01): # test values and see if other work uses this
    batch_size, num_nodes, feature_size = features.shape
    processed_tensor = []

    for i in range(batch_size):
        patient_data = np.array(features[i])

        random_int = np.random.randint(0, 2) # ~50% chance of adding noise
        if random_int == 0:
            # Skip noise for this patient
            processed_tensor.append(features[i])
            continue

        # Filter out rows that are all zeros
        non_zero_rows = np.any(patient_data != 0, axis=1)
        filtered_data = patient_data[non_zero_rows]

        # Add Gaussian noise to non-zero nodes
        noisy_data = torch.tensor(filtered_data) + torch.normal(mean, std, size=filtered_data.shape)

        # Create a padded tensor of shape (num_nodes, feature_size)
        padded_noisy_data = torch.zeros((num_nodes, feature_size))
        padded_noisy_data[:noisy_data.shape[0], :] = noisy_data

        # Append to the processed tensor list
        processed_tensor.append(padded_noisy_data)

    # Stack all processed patients back into a tensor of shape (batch_size, num_nodes, feature_size)
    processed_tensor = torch.stack(processed_tensor)

    return processed_tensor

def calibration_curve_and_distribution(all_labels, all_probs, label, results_path, Run, save=False):
    prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=10)
    plt.figure(figsize=(10, 5))

    # Plot calibration curve
    plt.plot(prob_pred, prob_true, marker='o', label='MIL-MLP')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('{} Calibration Curve'.format(label))
    plt.legend()
    if save:
        plt.savefig(results_path + f"/calibration_{label}_" + str(Run) + '.png')
    plt.show()

    # Plot distribution of predicted probabilities
    plt.hist(prob_pred, bins=10, edgecolor='k', alpha=0.7)
    plt.xlabel('Predicted probability')
    plt.ylabel('Frequency')
    plt.title('{} Distribution of Predicted Probabilities'.format(label))
    plt.tight_layout()
    if save:
        plt.savefig(results_path + f"/prob_distribution_{label}_" + str(Run) + '.png')
    plt.show()

    # Plot the roc curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} Receiver Operating Characteristic'.format(label))
    plt.legend(loc="lower right")
    if save:
        plt.savefig(results_path + f"/roc_curve_{label}_" + str(Run) + '.png')
    plt.show()
