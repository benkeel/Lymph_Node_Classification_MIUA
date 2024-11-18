import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, patch_features, mask):
        weights = self.attention_weights(patch_features)
        # Apply mask and ignore padded nodes when computing attention weights
        weights = weights.masked_fill(mask.bool(), float('-inf')) # make -inf so contribution to softmax is 0, then weights from non masked nodes and clinical features will sum to 1
        weights = F.softmax(weights, dim=1) # dim=0 for batchsize 1
        patient_representation = torch.sum(weights * patch_features, dim=1) # dim=0 for batchsize 1
        return patient_representation





###### Approach 1 #######
# inspired by Xia et al. 2024
class PatchMLP_probs_intensity(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, patch_dropout=0.2):
        super(PatchMLP_probs_intensity, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=patch_dropout)
        self.gelu1 = nn.GELU()

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.gelu1(x)

        x = self.fc2(x)
        x = self.sig(x)
        return x

class Patient_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, patient_dropout=0.2):
        super(Patient_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=patient_dropout)
        self.gelu1 = nn.GELU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.dropout2 = nn.Dropout(p=patient_dropout)
        self.gelu2 = nn.GELU()

        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.bn3 = nn.BatchNorm1d(hidden_dim//4)
        self.dropout3 = nn.Dropout(p=patient_dropout)
        self.gelu3 = nn.GELU()

        self.fc4 = nn.Linear(hidden_dim//4, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.gelu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.gelu2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.gelu3(x)

        x = self.fc4(x)
        x = self.sig(x)
        return x


class MLP_MIL_model2(nn.Module):
    def __init__(self, patch_input_dim, hyperparams, backbone_indicator=False, backbone=None, grad_cam=False):
        super(MLP_MIL_model2, self).__init__()

        num_classes = 1
        patch_hidden_dim = hyperparams["patch_hidden_dim"]
        patch_dropout = hyperparams["patch_dropout"]
        patch_output_dim = 1
        attention_dim = 1
        attention_indicator = hyperparams["attention_indicator"]
        patient_hidden_dim = hyperparams["patient_hidden_dim"]
        patient_dropout = hyperparams["patient_dropout"]
        max_node_slices = hyperparams["max_node_slices"]
        alpha = hyperparams["alpha"]
        device = hyperparams["device"]
        clinical_data_options = hyperparams["clinical_data_options"]
        clinical_features_length = 0
        if "size" in clinical_data_options:
            clinical_features_length += 3
        if "border" in clinical_data_options:
            clinical_features_length += 2
        if "patient" in clinical_data_options:
            clinical_features_length += 2
        if "T_stage" in clinical_data_options:
            clinical_features_length += 1



        self.patch_mlp = PatchMLP_probs_intensity(patch_input_dim, patch_hidden_dim, patch_output_dim, patch_dropout)
        self.attention = Attention(attention_dim)
        self.patient_mlp = Patient_MLP(max_node_slices + clinical_features_length, patient_hidden_dim, num_classes, patient_dropout)
        self.alpha = alpha
        self.clinical_data_options = clinical_data_options
        self.device = device
        self.attention_indicator = attention_indicator
        self.backbone_indicator = backbone_indicator
        if self.backbone_indicator:
            self.backbone = backbone
        self.grad_cam = grad_cam


    def forward(self, patches, clinical_data, number_of_nodes, labels, LN_features=None):
        if self.backbone_indicator:
            patches = torch.stack([self.backbone(patch) for patch in patches])
            patches = patches.squeeze(-1).squeeze(-1)
            if ("size" in self.clinical_data_options) or ("border" in self.clinical_data_options):
                patches = torch.cat((patches, LN_features), dim=-1)

        # Process each patch through the patch-level MLP
        patch_features = torch.stack([self.patch_mlp(patch) for patch in patches])

        # get initial max/attention based on image/intensity features
        max_vals = torch.zeros(len(patches), 1).to(self.device)
        if self.attention_indicator:
            attention_masks = torch.zeros(len(patches), len(patches[0])+len(clinical_data[0]), 1).to(self.device) #then save all masks and do the attention step outside of this for loop to preserve all the gradients
        for i, predictions in enumerate(patch_features):

            # Get the max value from only the real lymph nodes (not padded)
            max_node_idx = int(number_of_nodes[i]) # index of the last real node
            real_node_predictions = predictions[:max_node_idx]
            max_vals[i] = torch.max(real_node_predictions)

            # Create attention mask
            if self.attention_indicator:
                attention_mask = torch.tensor([0]*max_node_idx + [1]*(len(predictions)-max_node_idx) + [0]*len(clinical_data[i])).bool().to(self.device).unsqueeze(0).unsqueeze(2)
                attention_masks[i] = attention_mask

        # Combine patch features with clinical data
        all_predictions_with_clinical = torch.cat([patch_features, clinical_data.unsqueeze(2)], dim=1)

        # Apply attention mechanism
        if self.attention_indicator:
            attentions = self.attention(all_predictions_with_clinical, attention_masks)
        else:
            attentions = torch.zeros(len(patches), 1).to(self.device)
        all_predictions_with_clinical = torch.squeeze(all_predictions_with_clinical, 2)
        refined_LNM_predictions = self.patient_mlp(all_predictions_with_clinical)

        #output = 0.5*refined_LNM_predictions + 0.4*max_vals + 0.1*attentions
        if self.attention_indicator == True:
            output = 0.9*(self.alpha*refined_LNM_predictions + (1-self.alpha)*max_vals) + 0.1*attentions
            #output = refined_LNM_predictions
        if self.attention_indicator == False:
            output = self.alpha*refined_LNM_predictions + (1-self.alpha)*max_vals
            #output = refined_LNM_predictions

        #pre_output = concat(invlogit(refined_LNM_predictions), invlogit(max_vals), invlogit(attentions))
        #output = dense(pre_output, activation='sigmoid')
        if self.grad_cam:
            return output, max_vals, refined_LNM_predictions, patch_features


        return output, max_vals, attentions, refined_LNM_predictions






###### Approach 2 (old) #######
class PatchMLP_probs(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PatchMLP_probs, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim//2)
        self.bn1 = nn.BatchNorm1d(hidden_dim//2)
        self.dropout1 = nn.Dropout(p=0.35)
        self.gelu1 = nn.GELU()

        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=0.35)
        self.gelu2 = nn.GELU()

        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn3 = nn.BatchNorm1d(hidden_dim)
        # self.dropout3 = nn.Dropout(p=0.35)
        # self.gelu3 = nn.GELU()

        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        #x = x.unsqueeze(0) # hashtagged with batch size >1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.gelu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.gelu2(x)

        # x = self.fc3(x)
        # x = self.bn3(x)
        # x = self.dropout3(x)
        # x = self.gelu3(x)

        x = self.fc4(x)
        x = self.sig(x)
        #x = x.squeeze(0) # hashtagged with batch size >1
        return x



class MLP_MIL_model_simple(nn.Module):
    def __init__(self, patch_input_dim, hyperparams, backbone_indicator=False, backbone=None):
        super(MLP_MIL_model_simple, self).__init__()

        num_classes = 1
        patch_hidden_dim = hyperparams["patch_hidden_dim"]
        patch_output_dim = 1
        attention_dim = 1
        device = hyperparams["device"]
        self.patch_mlp = PatchMLP_probs(patch_input_dim, patch_hidden_dim, patch_output_dim)
        self.attention = Attention(attention_dim)
        self.device = device
        # self.classifier = nn.Linear(attention_dim, num_classes)
        self.backbone_indicator = backbone_indicator
        if self.backbone_indicator:
            self.backbone = backbone

        # Linear layer to combine clinical data (6) and attention + max val (2) #[now not incl patch features (30)]
        self.classifier = nn.Linear(6+2, num_classes)



    def forward(self, patches, clinical_data, number_of_nodes, labels):
        if self.backbone_indicator:
            patches = torch.stack([self.backbone(patch) for patch in patches])
            patches = patches.squeeze(-1).squeeze(-1)


        # Process each patch through the patch-level MLP
        patch_features = torch.stack([self.patch_mlp(patch) for patch in patches])

        max_vals = torch.zeros(len(patches), 1).to(self.device)
        attention_masks = torch.zeros(len(patches), len(patches[0])+len(clinical_data[0]), 1).to(self.device) #then save all masks and do the attention step outside of this for loop to preserve all the gradients? check if this needs to be done
        for i, predictions in enumerate(patch_features):

            # Get the max value from only the real lymph nodes (not padded)
            max_node_idx = int(number_of_nodes[i]) # index of the last real node
            real_node_predictions = predictions[:max_node_idx]
            max_vals[i] = torch.max(real_node_predictions)

            # Create attention mask
            attention_mask = torch.tensor([0]*max_node_idx + [1]*(len(predictions)-max_node_idx) + [0]*len(clinical_data[i])).bool().to(self.device).unsqueeze(0).unsqueeze(2)
            attention_masks[i] = attention_mask

        # Combine patch features with clinical data
        all_predictions_with_clinical = torch.cat([patch_features, clinical_data.unsqueeze(2)], dim=1)

        # Apply attention mechanism
        attentions = self.attention(all_predictions_with_clinical, attention_masks)

        #all_predictions_with_clinical = torch.cat([all_predictions_with_clinical, attentions.unsqueeze(2), max_vals.unsqueeze(2)], dim=1).squeeze(2)
        #classifications = self.classifier(all_predictions_with_clinical)

        # Classify LNM using logistic regression with patient level predictions (attention and max) combined with clinical data
        classification_features = torch.cat([attentions.unsqueeze(2), max_vals.unsqueeze(2), clinical_data.unsqueeze(2)], dim=1).squeeze(2)
        classifications = self.classifier(classification_features)
        classifications = torch.sigmoid(classifications)  # Apply sigmoid activation for binary classification


        output = 0.99*max_vals +  0.01*attentions #+ 0.15*classifications
        # idea: try get classification stats for each method to work out which proportions or use an attention mechanism to weight the outputs?

        return output, max_vals, attentions, classifications


