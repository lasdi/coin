#!/usr/bin/env python3

import torch
import torch.nn as nn

import sys
import random
import signal
import numpy as np

from brevitas.nn import QuantIdentity, QuantLinear
from brevitas_common import CommonWeightQuant, CommonActQuant
from brevitas_tensor_norm import TensorNorm

sys.path.append("../software_model")
sys.path.append("../backprop_experiments/binary_model/")
# from main_backprop import get_datasets

DROPOUT = 0.05
class FC_BNN(nn.Module):
    def __init__(self, num_inputs, num_classes, intermediate_layers, norm_min, norm_max):
        super(FC_BNN, self).__init__()

        self.norm_min = nn.Parameter(norm_min.type(torch.float32), requires_grad=False)
        self.norm_max = nn.Parameter(norm_max.type(torch.float32), requires_grad=False)

        self.features = nn.Sequential()
        self.features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=1))
        self.features.append(nn.Dropout(p=DROPOUT))

        layer_inps = num_inputs
        for layer_outps in intermediate_layers:
            print(layer_inps, layer_outps)
            self.features.append(QuantLinear(
                in_features=layer_inps,
                out_features=layer_outps,
                bias=False,
                weight_bit_width=1,
                weight_quant=CommonWeightQuant))
            self.features.append(nn.BatchNorm1d(num_features=layer_outps))
            self.features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=1))
            self.features.append(nn.Dropout(p=DROPOUT))
            layer_inps = layer_outps
        print(layer_inps, num_classes)
        self.features.append(QuantLinear(
                in_features=layer_inps,
                out_features=num_classes,
                bias=False,
                weight_bit_width=1,
                weight_quant=CommonWeightQuant))
        self.features.append(nn.BatchNorm1d(num_features=num_classes))
        # self.features.append(TensorNorm())

        for m in self.modules():
          if isinstance(m, QuantLinear):
            torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1).type(torch.float32)
        x = (x - self.norm_min) / (self.norm_max - self.norm_min)
        x = (2 * x) - 1
        return self.features.forward(x)

def run_inference(model, dset_loader):
    total = 0
    correct = 0
    device = next(model.parameters()).device
    for features, labels in dset_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return total, correct

Abort_Training = False
def sigint_handler(signum, frame):
    global Abort_Training
    if not Abort_Training:
        print("Will abort training at end of epoch")
        Abort_Training = True
    else:
        sys.exit("Quitting immediately on second SIGINT")

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=3e-3, device="cuda"):
    global Abort_Training
    Abort_Training = False
    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigint_handler)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        train_total = 0
        train_correct = 0
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            #features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
          
            _, predicted = torch.max(outputs.data, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            model.clip_weights(-1, 1)

        model.eval()
        val_total, val_correct = run_inference(model, val_loader)
        print(f"At end of epoch {epoch}: "\
                f"Train set: Correct: {train_correct}/{train_total} ({round(((100*train_correct)/train_total).item(), 3)}%); "\
                f"Validation set: Correct: {val_correct}/{val_total} ({round(((100*val_correct)/val_total).item(), 3)}%)")
        
        if Abort_Training:
            break
    
    model.eval()
    signal.signal(signal.SIGINT, old_handler)
    
    model = model.to("cpu")
    return model

def compute_model_size(num_inputs, num_classes, intermediate_layers):
    total_model_size = 0
    layer_inps = num_inputs
    for layer_outps in intermediate_layers:
        total_model_size += layer_inps * layer_outps
        layer_inps = layer_outps
    total_model_size += layer_inps * num_classes
    model_size_k = round(total_model_size / (2**13), 2)
    model_size_m = round(total_model_size / (2**23), 2)
    model_size_g = round(total_model_size / (2**33), 2)
    print(f"Total model size: {model_size_k} KiB / {model_size_m} MiB / {model_size_g} GiB")

def create_model(train_dataset, test_dataset, intermediate_layers, model_fname, num_epochs=25, learning_rate=4e-3):
    batch_size = 100
    num_workers = 4

    # train_dataset, test_dataset = get_datasets(ds_name)
    
    if isinstance(train_dataset, torch.utils.data.dataset.TensorDataset):
        num_inputs = train_dataset.tensors[0].shape[1]
        num_classes = (train_dataset.tensors[1].amax() + 1).item()
    elif isinstance(train_dataset.data[0], torch.Tensor):
        num_inputs = train_dataset.data[0].numel()
        num_classes = (train_dataset.targets.amax() + 1).item()
    elif isinstance(train_dataset.data[0], np.ndarray):
        num_inputs = train_dataset[0][0].numel()
        if hasattr(train_dataset, "labels"):
            num_classes = int(max(train_dataset.labels) + 1)
        if hasattr(train_dataset, "targets"):
            num_classes = int(max(train_dataset.targets) + 1)

    print(f"Num inputs/classes: {num_inputs}/{num_classes}")
    print(f"Batch size: {batch_size}")
    compute_model_size(num_inputs, num_classes, intermediate_layers)
    
    torch.manual_seed(0) # For reproducability
    # split_idx = int(len(train_dataset) * 0.9)
    # train_set, val_set = torch.utils.data.random_split(train_dataset, [split_idx, len(train_dataset)-split_idx])
    train_set = train_dataset
    test_set = test_dataset
        
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
   
    
    train_data = torch.stack(tuple(t[0].flatten() for t in train_set))
    model = FC_BNN(num_inputs, num_classes, intermediate_layers, train_data.amin(axis=0), train_data.amax(axis=0))

    model = train_model(model, train_loader, test_loader, num_epochs=num_epochs, learning_rate=learning_rate)
    
    test_total, test_correct = run_inference(model, test_loader)
    print(f"Test set: Correct: {test_correct}/{test_total} ({round(((100*test_correct)/test_total).item(), 3)}%)")
    torch.save({
        'state_dict': model.state_dict()
    }, model_fname)

    return model

if __name__ == "__main__":
    model_fname = "model.pt"
    if len(sys.argv) > 1:
        model_fname = sys.argv[1]
    #model = create_model("speech_commands", [128, 128], model_fname)
    # model = create_model("speech_commands", [768, 768, 768], model_fname)
    #model = create_model("UNSWNB15_multiclass", [128, 128, 128], model_fname)
    # model = create_model("mnist", [128, 128], model_fname)
    model = create_model("mnist", [], model_fname)
