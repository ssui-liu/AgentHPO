import torch
import torch.nn.functional as F

import json
import logging

from model import GCN


def setup_logger(name, file_name, mode='a'):
    logger = logging.getLogger(name)
    handler = logging.FileHandler(file_name, mode=mode)
    logger.addHandler(handler)
    return logger


def load_configs(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


def log_configs(configs, logger):
    config_log = "Configurations:\n" + "\n".join([f"\t{key}: {value}" for key, value in configs.items()])
    logger.info(config_log)


def initialize_model(configs, dataset, device):
    model = GCN(num_features=dataset.num_node_features,
                num_classes=dataset.num_classes,
                num_layers=configs['num_layers'],
                hidden_size=configs['hidden_size'],
                activation=configs['activation'],
                dropout=configs['dropout'])
    return model.to(device)


def choose_optimizer(model, configs):
    if configs['optimizer'] == "sgd":
        return torch.optim.SGD(model.parameters(),
                               lr=configs['learning_rate'],
                               weight_decay=configs['weight_decay'])
    else:
        return torch.optim.Adam(model.parameters(),
                                lr=configs['learning_rate'],
                                weight_decay=configs['weight_decay'])


def train_epoch(model, optimizer, data):
    running_corrects = 0
    total_train_samples = 0
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    _, preds = torch.max(out, 1)
    running_corrects += (preds[data.train_mask] == data.y[data.train_mask]).sum().item()
    total_train_samples += data.train_mask.sum().item()
    train_accuracy = running_corrects / total_train_samples
    return loss.item(), train_accuracy


def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()
        _, pred = out.max(dim=1)
        correct = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
        accuracy = correct / data.val_mask.sum().item()
    return loss, accuracy


def test_model(model, data):
    model.eval()
    with torch.no_grad():
        _, pred = model(data).max(dim=1)
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        accuracy = correct / data.test_mask.sum().item()

    return accuracy


def log_final_statistics(logger1, logger2, accuracy, total_training_time, trajectories):
    log_items = [
        "Training Trajectory:",
        "Epoch: [" + ", ".join(map(str, trajectories['epochs'])) + "]",
        "Train Loss: [" + ", ".join(map(str, trajectories['train_losses'])) + "]",
        "Train Acc: [" + ", ".join(map(str, trajectories['train_accs'])) + "]",
        "Val Loss: [" + ", ".join(map(str, trajectories['val_losses'])) + "]",
        "Val Acc: [" + ", ".join(map(str, trajectories['val_accs'])) + "]",
        f"Total Training Time: {total_training_time:.3f}s",
        f"Final Validation Accuracy: {accuracy:.4f}%"
    ]

    for item in log_items:
        logger1.info(item)
        logger2.info(item)
