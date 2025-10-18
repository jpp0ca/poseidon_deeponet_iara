import os

# SOLUÇÃO: Controla o paralelismo para evitar conflitos e o aviso do OpenBLAS.
# Força as bibliotecas a usarem apenas um thread, o que resolve o problema 
# de paralelismo aninhado.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
# Adicionando MLP e Trainer aos imports
from poseidon.models.mlp import MLP
from poseidon.models.cnn import CNN
from poseidon.models.deeponet import DeepONet
from poseidon.training import Trainer, calculate_class_weights, DeepONetTrainer
from poseidon.io.iara.offline import load_sonar_from_csv, Target, load_processed_data
from poseidon.signal_poseidon.passivesonar import lofar
from poseidon.signal_poseidon.utils import resample
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import wandb
from poseidon.data_handling import CustomDataloader, LoroCV, DeepOnetDataLoader
from poseidon.visualization import plot_lofargram, plot_tsne_embeddings, palette
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from dotwiz import DotWiz
import matplotlib
matplotlib.use('agg')


PATHS = {"dataset": {
            "name": "iara",
            "metadata_path": "../data/iara.csv",
            "raw_data_path" : "/home/iara/",
            "cache_path" : "./data/iara_cache"
        },
        "dc_filter": ['D', 'H']}

def save_embeddings_and_targets(config, embeddings, all_targets, results_path):
    """
    Save each embedding and target array into separate .npy files.

    Parameters:
        embeddings (list of arrays): List of embedding arrays from each fold.
        all_targets (list of arrays): List of target arrays from each fold.
        results_path (Path): Path object to the directory where files will be saved.
    """
    for i, (embedding, target) in enumerate(zip(embeddings, all_targets)):
        embedding = np.array(embedding)
        target = np.array(target)

        np.save(results_path / "data" / f"embeddings_fold_{config.fold}.npy", embedding)
        np.save(results_path / "data" / f"all_targets_fold_{config.fold}.npy", target)

def lofar_fn(x):
    signal = resample(x['signal'], x['fs'], 16000)
    fs = 16000
    # x is a dict with keys 'signal' and 'fs'
    # x['signal'] is the audio signal, x['fs'] is the sampling frequency
    return lofar(signal, fs, n_pts_fft=1024, n_overlap=0,
                  spectrum_bins_left=512)

def load_data(config):
    csv_path = config.dataset.metadata_path
    iara_data_root_path = config.dataset.raw_data_path
    iara_raw = load_sonar_from_csv(csv_path, data_root_path=iara_data_root_path, target_column="Ship Length Class", data_collection_filter=config.dc_filter)

    cache_dir = config.dataset.cache_path
    Path(cache_dir).mkdir(parents=True, exist_ok=True)  
    
    iara_raw.process_and_cache(fn=lofar_fn, max_workers=8, cache_path=cache_dir)

    iara_spectrogram = load_processed_data(cache_dir)

    print("=" * 75)
    print("Completed Data Preprocessing with the Following Configuration:")
    print(f" - FFT Points               : {1024}")
    print(f" - Window Overlap           : {0}")
    print(f" - Decimation Rate          : {3}")
    print(f" - Final Sampling Frequency : {16000}")
    print("=" * 75)

    return iara_spectrogram

def model_select(config, branch_net = None):
    window_size = config.window_size
    
    if config.model_name == "DeepONet-MLP-MLP":
        return lambda input_size, coords: DeepONet(branch_net= MLP(input_shape=input_size,
                                                           hidden_channels=config.hidden_channels, 
                                                           n_targets=config.embedding_dim, 
                                                           dropout=config.dropout),
                                           trunk_net= MLP(input_shape=coords,
                                                           hidden_channels=config.hidden_channels, 
                                                           n_targets=config.embedding_dim, 
                                                           dropout=config.dropout),
                                           class_head= MLP(input_shape=32,
                                                           hidden_channels=config.hidden_channels,
                                                           n_targets=4,
                                                           dropout=config.dropout))
    elif config.model_name == "DeepONet-CNN-MLP":
        return lambda input_size, coords: DeepONet(branch_net= CNN(input_shape=input_size,
                                                                    conv_n_neurons=config.conv_n_neurons,
                                                                    conv_activation=torch.nn.PReLU,
                                                                    conv_pooling=torch.nn.MaxPool2d,
                                                                    conv_pooling_size=config.conv_pooling_size,
                                                                    conv_dropout=config.conv_dropout,
                                                                    batch_norm=torch.nn.BatchNorm2d,
                                                                    kernel_size=config.kernel_size,
                                                                    has_class_head=False,
                                                                    hidden_channels=config.classification_n_neurons,
                                                                    n_targets=config.embedding_dim,
                                                                    dropout=config.classification_dropout),
                                           trunk_net= MLP(input_shape=coords,
                                                           hidden_channels=config.hidden_channels, 
                                                           n_targets=config.embedding_dim, 
                                                           dropout=config.dropout),
                                           class_head= MLP(input_shape=16384,
                                                           hidden_channels=config.hidden_channels,
                                                           n_targets=4,
                                                           dropout=config.dropout))
    elif config.model_name == "MLP":
        return lambda input_size, coords: MLP(input_shape=input_size, hidden_channels=config.hidden_channels, n_targets=4, dropout=config.dropout)
    
    elif config.model_name == "CNN":
        return lambda input_size: CNN(input_shape=input_size,
                                      conv_n_neurons=config.conv_n_neurons,
                                      conv_activation=torch.nn.PReLU,
                                      conv_pooling=torch.nn.MaxPool2d,
                                      conv_pooling_size=config.conv_pooling_size,
                                      conv_dropout=config.conv_dropout,
                                      batch_norm=torch.nn.BatchNorm2d,
                                      kernel_size=config.kernel_size,
                                      has_class_head=True,
                                      hidden_channels=config.classification_n_neurons,
                                      n_targets=4,
                                      dropout=config.classification_dropout)
    else:
        raise ValueError(f"Model name {config.model_name} not recognized.")

def run_experiment(config, lofar_data, results_path, device):
    # Initialize the model, optimizer, and criterion
    alpha = config.alpha if hasattr(config, 'alpha') else None
    window_size = config.window_size
    
    non_multitask_models_list = ["MLP", "DeepONet-MLP-MLP", "DeepONet-CNN-MLP"]

    if window_size is None:
        overlap = None
    elif window_size == 16:
        overlap = 14
    elif window_size == 32:
        overlap = 28
    else:
        raise ValueError(f"Window size {window_size} not recognized.")

    model_builder = model_select(config)
    # Perform cross-validation using LoroCV
    accuracies = []
    embeddings = []
    all_targets    = []
    lorocv_no_window = LoroCV(n_splits=5, window_size=window_size, overlap=overlap, random_seed=42)

    fold = config.fold
    for i, (X_train, y_train, X_test, y_test, coords_train, coords_test) in enumerate(lorocv_no_window.split(lofar_data)):
        if i != fold:
            continue
        # Compute class weights for loss balancing
        class_weights = calculate_class_weights(y_train).to(device)
        
        if config.model_name in ["CNN", "DeepONet-CNN-MLP"]:
            X_train = np.expand_dims(X_train, axis=1) # Adiciona a dimensão do canal
            X_test = np.expand_dims(X_test, axis=1)
        
        # Create DataLoader instances for the fold
        is2d = window_size is not None and config.model_name != "MLP"
        if config.model_name in ["DeepONet-MLP-MLP", "DeepONet-CNN-MLP"]:
            train_dataset_fold = DeepOnetDataLoader(X_train, y_train, coords_train, device=device)
            test_dataset_fold = DeepOnetDataLoader(X_test, y_test, coords_test, device=device)
        else:
            train_dataset_fold = CustomDataloader(X_train, y_train, is2d=is2d, device=device)
            test_dataset_fold = CustomDataloader(X_test, y_test, is2d=is2d, device=device)
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=32, shuffle=True, drop_last=True)
        test_loader_fold = DataLoader(test_dataset_fold, batch_size=32, shuffle=False, drop_last=True)

        input_size = X_train.shape[1:] if config.model_name in ["CNN", "DeepONet-CNN-MLP"] else X_train.shape[1]
        coords_size = coords_train.shape[1]
        model_fold = model_builder(input_size, coords_size).to(device)
        optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_fold, gamma=0.93)
        clf_criterion_fold = torch.nn.CrossEntropyLoss(weight=class_weights)

        # if config.model_name in non_multitask_models_list:
        
        trainer_fold = Trainer(model_fold, optimizer_fold, scheduler, clf_criterion_fold,
                                   num_epochs=100, verbose=True, wandb_logging=True)
        
        if config.model_name in ["DeepONet-MLP-MLP", "DeepONet-CNN-MLP"]:
            trainer_fold = DeepONetTrainer(model_fold, optimizer_fold, scheduler, clf_criterion_fold,
                                    num_epochs=100, verbose=True, wandb_logging=True)
            
        trainer_fold.train(train_loader_fold, test_loader_fold, patience=10)
        
        _, accuracy, precision, recall, f1, roc_auc, y_pred, y_target = trainer_fold.evaluate(test_loader_fold)
        wandb.log({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })
        
        np.save(results_path / "data" / f"predictions_fold_{fold}.npy", y_pred)
        np.save(results_path / "data" / f"targets_fold_{fold}.npy", y_target)
                    
        fold_embeddings, fold_targets, fold_scores = trainer_fold.evaluate_embeddings(test_loader_fold)
        embeddings.append(fold_embeddings)
        all_targets.append(fold_targets)

        accuracies.append(accuracy)

        wandb.log(fold_scores)
        
        fig, ax = plt.subplots(figsize=(12, 12/ 1.618))
        plot_tsne_embeddings(ax, fold_embeddings, fold_targets, palette=palette)
        plot_name = f"t-SNE_embeddings_fold_{i}"
        fig.savefig(results_path / "plots" / "png" / f"{plot_name}.png", bbox_inches='tight', dpi=300)
        fig.savefig(results_path / "plots" / "svg" / f"{plot_name}.svg", bbox_inches='tight')

        wandb.log({"t-SNE plot": wandb.Image(fig)})
        plt.close(fig)
        
        continue
        
        # else:
        #     rec_criterion_fold = torch.nn.MSELoss()
        #     trainer_fold = MultitaskTrainer(model_fold, optimizer_fold, scheduler,
        #                                     clf_criterion_fold,
        #                                     rec_criterion_fold,
        #                                     alpha=alpha,
        #                                     num_epochs=100, verbose=True, wandb_logging=True)
        #     trainer_fold.train(train_loader_fold, test_loader_fold, patience=10)
            
        #     # Evaluate the model on the test fold
        #     _, accuracy, precision, recall, f1, sp, roc_auc = trainer_fold.evaluate(test_loader_fold)
        #     wandb.log({
        #         "accuracy": accuracy,
        #         "precision": precision,
        #         "recall": recall,
        #         "f1_score": f1,
        #         "sp_index": sp,
        #         "roc_auc": roc_auc
        #     })

        #     # Evaluate the embeddings
        #     fold_embeddings, fold_targets, fold_scores = trainer_fold.evaluate_embeddings(test_loader_fold)
        #     embeddings.append(fold_embeddings)
        #     all_targets.append(fold_targets)

        #     accuracies.append(accuracy)

        #     wandb.log(fold_scores)

        #     fig, ax = plt.subplots(figsize=(12, 12/ 1.618))
        #     plot_tsne_embeddings(ax, fold_embeddings, fold_targets, palette=palette)
        #     plot_name = f"t-SNE_embeddings_fold_{i}"
        #     fig.savefig(results_path / "plots" / "png" / f"{plot_name}.png", bbox_inches='tight', dpi=300)
        #     fig.savefig(results_path / "plots" / "svg" / f"{plot_name}.svg", bbox_inches='tight')

        #     wandb.log({"t-SNE plot": wandb.Image(fig)})
        #     plt.close(fig)

        #     continuous_eval = trainer_fold.evaluate_embeddings_continuity(fold_embeddings, fold_targets)
        #     continuous_embeddings = {i_cls: continuous_eval[i_cls]["class_embeddings"] for i_cls in continuous_eval.keys()}

        #     fig, axes = plt.subplots(4, 2, figsize=(8, 8/ 1.618))
        #     for i_cls, ce_scores in continuous_eval.items():
        #         spearman_correlation = ce_scores["scores"]["spearman"]["correlation"]
        #         distances_from_start = ce_scores["scores"]["spearman"]["distances_from_start"]
        #         time_ranks = ce_scores["scores"]["spearman"]["time_ranks"]

        #         local_consistency = ce_scores["scores"]["local_consistency"]["local_consistency"]
        #         step_lengths = ce_scores["scores"]["local_consistency"]["step_lengths"]

        #         axes[i_cls, 0].plot(time_ranks, distances_from_start)
        #         axes[i_cls, 1].plot(time_ranks, step_lengths)

        #     plot_name = f"continuity_measures_fold{i}"
        #     fig.savefig(results_path / "plots" / "png" / f"{plot_name}.png", bbox_inches='tight', dpi=300)
        #     fig.savefig(results_path / "plots" / "svg" / f"{plot_name}.svg", bbox_inches='tight')

        #     wandb.log({"Continuity Measures": wandb.Image(fig)})
        #     plt.close(fig)

        #     fig, axes = plt.subplots(2, 2, figsize=(8, 8/ 1.618), sharex=True, sharey=True)
        #     axf = axes.flat
        #     for i_cls, (trgt, cls_embeddings) in enumerate(continuous_embeddings.items()):
        #         num_windows = len(cls_embeddings)
        #         indices = np.array(list(range(num_windows)))/num_windows
        #         axf[i_cls].scatter(cls_embeddings[:, 0], cls_embeddings[:, 1], s=1,
        #                        c=indices, cmap='inferno', alpha=0.7)
        #         axf[i_cls].set_xlabel('t-SNE Dimension 1', fontsize=8)
        #         axf[i_cls].set_ylabel('t-SNE Dimension 2', fontsize=8)
        #         axf[i_cls].grid(True, linestyle='--', alpha=0.5)
        #         axf[i_cls].axhline(0, color='black', linewidth=0.5)
        #         axf[i_cls].axvline(0, color='black', linewidth=0.5)

        #     norm = plt.Normalize(vmin=0, vmax=1)
        #     sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
        #     cbar = fig.colorbar(sm, ax=axf, orientation='vertical', pad=0.1, fraction=0.02)
        #     cbar.set_label('Normalized Time Progression')

        #     plot_name = f"t-SNE_embeddings-CONTINUITY_fold_{i}"
        #     fig.savefig(results_path / "plots" / "png" / f"{plot_name}.png", bbox_inches='tight', dpi=300)
        #     fig.savefig(results_path / "plots" / "svg" / f"{plot_name}.svg", bbox_inches='tight')

        #     wandb.log({"t-SNE continuity plot": wandb.Image(fig)})
        #     plt.close(fig)

    return embeddings, all_targets



def make_hp_name(config):
    alpha = config.alpha if hasattr(config, 'alpha') else 'na'
    latent_dim_size = config.latent_dim_size if hasattr(config, 'latent_dim_size') else 'na'
    output_size = config.output_size if hasattr(config, 'output_size') else 'na'
    window_size = config.window_size
    learning_rate = config.learning_rate
    

    if config.model_name == "MLP":
        hidden_str = '_'.join(map(str, config.hidden_channels))
        return f"hidden_{hidden_str}_dropout_{config.dropout}_lr_{learning_rate}"
    if config.model_name == "DeepONet-MLP":
         hidden_str = '_'.join(map(str, config.hidden_channels))
         return f"hidden_{hidden_str}_dropout_{config.dropout}_lr_{learning_rate}_embedding_{config.embedding_dim}"
    elif config.model_name == "MultitaskAutoencoder":
        return f"alpha_{alpha}_latent_{latent_dim_size}_window_{window_size}_lr_{learning_rate}"
    elif config.model_name == "ConvAutoencoderMultitask":
        return f"alpha_{alpha}_latent_{latent_dim_size}_output_{output_size}_window_{window_size}_lr_{learning_rate}"
    elif config.model_name == "MultitaskUNet":
        return f"alpha_{alpha}_latent_{latent_dim_size}_output_{output_size}_window_{window_size}_lr_{learning_rate}"
    elif config.model_name == "CNN":
        return f"conv_neurons_{config.conv_n_neurons}_pooling_{config.conv_pooling_size}_dropout_{config.conv_dropout}_kernel_{config.kernel_size}_class_neurons_{config.classification_n_neurons}_class_dropout_{config.classification_dropout}_lr_{learning_rate}"
    elif config.model_name == "CKAN":
        return f"window_{window_size}_grid_{config.grid_size}_dropout_{config.dropout}_lr_{learning_rate}"
    elif config.model_name == "CNN-BIGBOSS":
        return f"conv_neurons_{config.conv_n_neurons}_pooling_{config.conv_pooling_size}_dropout_{config.conv_dropout}_kernel_{config.kernel_size}_class_neurons_{config.classification_n_neurons}_class_dropout_{config.classification_dropout}_lr_{learning_rate}"
    else:
        raise ValueError(f"Model name {config.model_name} not recognized.")

def has_been_run(hash):
    hash_file = "config_hashes.txt"
    if not os.path.exists(hash_file):
        return False
    with open(hash_file, "r") as file:
        existing_hashes = file.read().split()
    return hash in existing_hashes

def store_hash(hash):
    with open("config_hashes.txt", "a") as file:
        file.write(hash + "\n")

def sweep_experiment(project_name, run_name):
    wandb.init(project=project_name)
    config = wandb.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = config.model_name
    hp_name = make_hp_name(config)
    fold = config.fold

    model_id = f"{model_name}_{hp_name}"
    model_hash = f"Fold_{fold}_{model_id}"

    if has_been_run(model_hash):
        print("Configuration has already been run. Skipping...")
        wandb.log({"duplicate": True})
        return
    config.model_id = model_id

    if args.debug:
        results_path = Path(f"./results/debug/{run_name}/{hp_name}")
    else:
        results_path = Path(f"./results/production/{run_name}/{hp_name}")

    (results_path / "plots" / "svg").mkdir(parents=True, exist_ok=True)
    (results_path / "plots" / "png").mkdir(parents=True, exist_ok=True)
    (results_path / "data").mkdir(parents=True, exist_ok=True)
    
    cons_paths = DotWiz(PATHS)

    lofar_data = load_data(cons_paths)

    embeddings, all_targets = run_experiment(config, lofar_data, results_path, device)

    if config.model_name != "MLP":
        save_embeddings_and_targets(config, embeddings, all_targets, results_path)

    store_hash(model_hash)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the model.')
    parser.add_argument('--config', type=str, default='config', help='Path to the configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # lofar_data, _, _ = load_data()

    config_file = f"./configs/{args.config}.json"
    with open(config_file, 'r') as f:
        sweep_configuration = json.load(f)

    if args.debug:
        project_name = f'{args.config}-debug-v6'
    else:
        project_name = f'{args.config}-v6'
    sweep_configuration['name'] = f"{project_name}-sweep"

    sweep_id = wandb.sweep(sweep_configuration, project=project_name)

    wandb.agent(sweep_id, function=lambda : sweep_experiment(project_name, project_name))
