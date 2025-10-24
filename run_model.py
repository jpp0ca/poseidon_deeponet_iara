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
from poseidon.training import Trainer, DeepONetTrainer
from poseidon.io.iara.offline import load_sonar_from_csv, Target, load_processed_data
from poseidon.signal_poseidon.passivesonar import lofar
from poseidon.signal_poseidon.utils import resample
from poseidon.model_selection import SonarCrossValidator
from poseidon.dataset.dataset import SonarRunDataset, SonarRunDeepONetDataset
from poseidon.utils import calculate_class_weights
from pathlib import Path
import numpy as np
import pandas as pd
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
import datetime as dt


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

def model_select(config, branch_net = None):
    
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
        return lambda input_size, coords: CNN(input_shape=input_size,
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

def run_experiment(config, results_path, device):
    # Initialize the model, optimizer, and criterion
    alpha = config.alpha if hasattr(config, 'alpha') else None

    csv_path = PATHS["dataset"]["metadata_path"]
    metadata_df = pd.read_csv(csv_path)
    metadata_df['Length'] = metadata_df['Length'].apply(lambda x: np.nan if x == ' - ' else float(x))
    metadata_df['Ship Length Class'] = metadata_df['Length'].apply(Target.classify_value)
    if PATHS["dc_filter"]:
        metadata_df = metadata_df[metadata_df['Dataset'].isin(PATHS["dc_filter"])]
    metadata_df.reset_index(drop=True, inplace=True)

    # Inicializa o validador cruzado com estratificação
    cross_validator = SonarCrossValidator(
        metadata_df=metadata_df,
        target_column='Ship Length Class',
        stratify_columns=['Ship Length Class', 'Dataset'],
        n_splits=5,
        random_state=42
    )
    
    labels = metadata_df['Ship Length Class']

    # Pega os dados (nomes de arquivo e labels) para o fold específico
    fold = config.fold
    cache_dir = PATHS["dataset"]["cache_path"]
    train_data_files, test_data_files = cross_validator.get_fold_data(fold, cache_dir)
    
    # Define os parâmetros de janelamento
    window_size = config.window_size
    is2d = config.model_name in ["CNN", "DeepONet-CNN-MLP"]
    if window_size is None or window_size == 1:
        overlap = 0
        is2d = False
    elif window_size == 16:
        overlap = 14
    elif window_size == 32:
        overlap = 28
    else:
        raise ValueError(f"Window size {window_size} não reconhecido.")

    # Seleciona o Dataset correto com base no nome do modelo
    if config.model_name in ["DeepONet-MLP-MLP", "DeepONet-CNN-MLP"]:
        print("Usando o DataLoader para DeepONet...")
        train_dataset = SonarRunDeepONetDataset(train_data_files, window_size=window_size, overlap=overlap, is2d=is2d)
        test_dataset = SonarRunDeepONetDataset(test_data_files, window_size=window_size, overlap=overlap, is2d=is2d)
    else:
        print("Usando o DataLoader padrão...")
        train_dataset = SonarRunDataset(train_data_files, window_size=window_size, overlap=overlap, is2d=is2d)
        test_dataset = SonarRunDataset(test_data_files, window_size=window_size, overlap=overlap, is2d=is2d)

    print("Passando pelo DataLoader do torch")
    train_loader_fold = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader_fold = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
    
    coords_size = None
    input_size = None
        
    first_batch = next(iter(train_loader_fold))
    sample_batch_inputs = first_batch[0] # O primeiro item do batch são sempre as features

    if config.model_name in ["DeepONet-MLP-MLP", "DeepONet-CNN-MLP"]:
        # Para o DeepONet, as features são uma tupla: (dados, coordenadas)
        sample_x, sample_coords = sample_batch_inputs
        input_size = sample_x.shape[1:]
        coords_size = sample_coords.shape[1:]
    else:
        # Para modelos padrão, as features são apenas o tensor de dados
        input_size = sample_batch_inputs.shape[1:]
    
    print("Definindo o modelo")
    model_builder = model_select(config)
    model_fold = model_builder(input_size, coords_size).to(device)
    
    print("Definindo optm")
    optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_fold, gamma=0.93)
    
    print("Definindo Pesos")
    class_weights = calculate_class_weights(labels, device)
    print(type(class_weights))
    
    class_weights = torch.tensor(class_weights, device= device)
    
    print ("Definindo Loss")
    clf_criterion_fold = torch.nn.CrossEntropyLoss(weight=class_weights)

    # if config.model_name in non_multitask_models_list:
    
    print("Inputando o trainer")
    trainer_fold = Trainer(model_fold, optimizer_fold, scheduler, clf_criterion_fold,
                                num_epochs=100, verbose=True, wandb_logging=True)
    
    if config.model_name in ["DeepONet-MLP-MLP", "DeepONet-CNN-MLP"]:
        trainer_fold = DeepONetTrainer(model_fold, optimizer_fold, scheduler, clf_criterion_fold,
                                num_epochs=100, verbose=True, wandb_logging=True)
    
    print("Iniciando o treinamento")
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

    embeddings, all_targets = run_experiment(config, results_path, device)

    if config.model_name != "MLP":
        save_embeddings_and_targets(config, embeddings, all_targets, results_path)

    store_hash(model_hash)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the model.')
    parser.add_argument('--config', type=str, default='config', help='Path to the configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    now_str = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # lofar_data, _, _ = load_data()

    config_file = f"./configs/{args.config}.json"
    with open(config_file, 'r') as f:
        sweep_configuration = json.load(f)

    if args.debug:
        project_name = f'{args.config}-debug-{now_str}'
    else:
        project_name = f'{args.config}-{now_str}'
    sweep_configuration['name'] = f"{project_name}-sweep"

    sweep_id = wandb.sweep(sweep_configuration, project=project_name)

    wandb.agent(sweep_id, function=lambda : sweep_experiment(project_name, project_name))
