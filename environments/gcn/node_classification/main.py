import traceback

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import json
import logging
import time
import os
import psutil
from model import GCN
from utils import (setup_logger, load_configs, initialize_model, log_configs, test_model,
                   choose_optimizer, train_epoch, evaluate_model, log_final_statistics)


def main():
    logging.basicConfig(level=logging.INFO)
    logger1 = setup_logger('Append Logger', 'append.log', mode='a')
    logger2 = setup_logger('Refresh Logger', 'refresh.log', mode='w')
    log_interval = 5
    try:
        configs = load_configs('configs_changed.json')
        # Log configurations
        log_configs(configs, logger1)
        log_configs(configs, logger2)
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = initialize_model(configs, dataset, device)
        optimizer = choose_optimizer(model, configs)
        data = dataset[0].to(device)

        trajectories = {
            'epochs': [],
            'train_losses': [],
            'train_accs': [],
            'val_losses': [],
            'val_accs': []
        }

        total_memory_usage = 0
        start_time = time.time()
        best_val_acc = 0
        patience = 10
        for epoch in range(configs['num_epochs']):
            train_loss, train_accuracy = train_epoch(model, optimizer, data)
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
            total_memory_usage += memory_usage
            val_loss, val_accuracy = evaluate_model(model, data)
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience = 10
            else:
                patience -= 1
            if patience == 0:
                break

            if epoch % log_interval == 0:

                trajectories['epochs'].append(epoch)
                trajectories['train_losses'].append(round(train_loss, 4))
                trajectories['train_accs'].append(round(train_accuracy, 4))
                trajectories['val_losses'].append(round(val_loss, 4))
                trajectories['val_accs'].append(round(val_accuracy, 4))

        end_time = time.time()
        # Additional final logging after all epochs
        total_training_time = end_time - start_time
        log_final_statistics(logger1, logger2, best_val_acc, total_training_time, trajectories)

    except Exception as e:
        error_log = f"Error occurred: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger1.error(error_log)
        logger2.error(error_log)
        print(error_log)

    logger1.info("Finished Training\n")
    logger2.info("Finished Training\n")
    print('Finished Training')


if __name__ == "__main__":
    main()
