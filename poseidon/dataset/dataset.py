import torch
from torch.utils.data import Dataset
import random
import numpy as np

class SonarRunDataset(Dataset):
    """
    A lazy-loading Dataset that uses a stateless SpectrogramLoader.
    This version is robust against memory accumulation in DataLoader workers.
    """
    def __init__(self, run_data, window_size, overlap, is2d=False):
        """
        Args:
            run_data (list): A list of tuples, where each tuple is 
                             (SpectrogramLoader, class_label_int).
        """
        self.run_data = run_data
        self.window_size = window_size
        self.stride = window_size - overlap
        self.is2d = is2d
        
        print("Building window map for SonarRunDataset...")
        self.window_map = []
        for run_idx, (loader, _) in enumerate(self.run_data):
            # Getting shape is now a property of the loader
            sxx_len = loader.shape[0]
            
            max_start_idx = sxx_len - self.window_size + 1
            for start_idx in range(0, max_start_idx, self.stride):
                self.window_map.append((run_idx, start_idx))
        print(f"Map built. Total windows available: {len(self.window_map)}")

    def __len__(self):
        return len(self.window_map)

    def __getitem__(self, idx):
        run_idx, start_idx = self.window_map[idx]
        loader, label = self.run_data[run_idx]
        
        # Load the spectrogram data from disk (stateless call)
        sxx = loader['sxx']
        
        window = torch.from_numpy(sxx[start_idx : start_idx + self.window_size]).float()
        
        if self.is2d:
            window = window.unsqueeze(0)
            
        return window, torch.tensor(label, dtype=torch.long)



class SonarRunPairDataset(Dataset):
    """
    A lazy-loading, memory-safe version of TimeSeriesPairDataset.

    It builds a map of all possible window pairs from a list of runs but only
    loads the full spectrogram data from disk when a specific item is requested
    by the DataLoader.
    """
    def __init__(self, run_data, window_size, overlap, is2d=False, max_offset: int = 1):
        """
        Args:
            run_data (list): A list of tuples, where each tuple is 
                             (SpectrogramLoader, class_label_int).
            window_size (int): The size of each window (number of stacked spectrums).
            overlap (int): The overlap between consecutive windows.
            is2d (bool): If True, adds a channel dimension to the window tensor.
            max_offset (int): The maximum number of steps between the start of the
                              first and second window in the pair.
        """
        self.run_data = run_data
        self.window_size = window_size
        self.stride = window_size - overlap
        self.is2d = is2d
        self.max_offset = max_offset

        print("Building window map for SonarRunPairDataset...")
        self.window_map = []
        for run_idx, (loader, _) in enumerate(self.run_data):
            # Getting shape is now a property of the loader. This is memory-safe.
            sxx_len = loader.shape[0]
            
            # The maximum starting index for a valid window PAIR
            max_start_idx = sxx_len - self.window_size - self.max_offset + 1
            
            for start_idx in range(0, max_start_idx, self.stride):
                self.window_map.append((run_idx, start_idx))
                
        print(f"Map built. Total window pairs available: {len(self.window_map)}")

    def __len__(self):
        return len(self.window_map)

    def __getitem__(self, idx):
        # 1. Look up which run and start index this pair corresponds to
        run_idx, start_idx_t = self.window_map[idx]
        
        # 2. Get the stateless loader and label for that run
        loader, label = self.run_data[run_idx]
        
        # 3. Load the full spectrogram data from disk (just-in-time)
        # Memory for 'sxx' will be freed when this method returns.
        sxx = loader['sxx']
        
        # 4. Slice out the first window (x_t)
        window_t = sxx[start_idx_t : start_idx_t + self.window_size]
        
        # 5. Get a random offset for the second window
        random_offset = random.randint(1, self.max_offset)
        start_idx_t_plus_offset = start_idx_t + random_offset
        
        # 6. Slice out the second window
        window_t_plus_offset = sxx[start_idx_t_plus_offset : start_idx_t_plus_offset + self.window_size]
        
        # 7. Convert to tensors
        window_t = torch.from_numpy(window_t).float()
        window_t_plus_offset = torch.from_numpy(window_t_plus_offset).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.is2d:
            window_t = window_t.unsqueeze(0)
            window_t_plus_offset = window_t_plus_offset.unsqueeze(0)
            
        return window_t, window_t_plus_offset, label_tensor
    

class SonarRunDeepONetDataset(SonarRunDataset):
    """
    Dataset customizado para o DeepONet que herda de SonarRunDataset.

    Além de carregar e janelar os dados, esta classe gera as coordenadas
    normalizadas (t, x) para cada janela, que são necessárias para o
    branch net do DeepONet.
    """
    def __init__(self, run_data, window_size, overlap=0, is2d=False):
        # Reutiliza o construtor da classe pai para preparar os dados
        super().__init__(run_data, window_size, overlap, is2d)
        self.coords = None

    def __getitem__(self, idx):
        # Pega a janela do espectrograma e o label usando a lógica da classe pai
        spectrogram_window, label = super().__getitem__(idx)

        # Agora, gera as coordenadas para esta janela
        if self.is2d:
            # Caso 2D (ex: para CNN-DeepONet)
            # A entrada é (C, H, W), mas as coordenadas são sobre H, W
            # Supondo que o shape seja (C, H, W) -> pegamos H e W
            _, height, width = spectrogram_window.shape
            
            # Cria vetores de coordenadas para os eixos de tempo e frequência
            grid_t = torch.linspace(-1, 1, steps=width)
            grid_f = torch.linspace(-1, 1, steps=height)
            
            # Cria a grade (meshgrid) e empilha para ter pares (t, f)
            mesh_t, mesh_f = torch.meshgrid(grid_t, grid_f, indexing='xy')
            coords = torch.stack((mesh_t.flatten(), mesh_f.flatten()), dim=1)

        else:
            # Caso 1D (ex: para MLP-DeepONet)
            # A entrada é (L,)
            length = spectrogram_window.shape[0]
            coords = torch.linspace(-1, 1, steps=length).unsqueeze(-1) # Shape (L, 1)
        
        self.coords = coords

        # O modelo DeepONet espera a entrada como uma tupla (função, coordenadas)
        # Então, o DataLoader deve retornar ((espectrograma, coords), label)
        return (spectrogram_window.float(), coords.float()), torch.tensor(label, dtype=torch.long)
