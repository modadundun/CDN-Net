# 文件名: load_cdn_v2.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class CdnDatasetV2(Dataset):
    """为 CDN 模型 V2 加载预处理后的数据 (PPG, Context, Target)"""
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self._load_data()

    def _load_data(self):
        print(f"Loading CDN V2 data for split: {self.split} from {self.data_dir}")
        try:
            self.X_ppg = np.load(os.path.join(self.data_dir, f'X_ppg_{self.split}.npy')).astype(np.float32)
            self.X_context = np.load(os.path.join(self.data_dir, f'X_context_{self.split}.npy')).astype(np.float32)
            self.y = np.load(os.path.join(self.data_dir, f'y_{self.split}.npy')).astype(np.float32)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: Missing .npy files in '{self.data_dir}'. Run preprocessing first. ({e})")

        print(f"  PPG shape: {self.X_ppg.shape}, Context shape: {self.X_context.shape}, Target shape: {self.y.shape}")
        assert len(self.X_ppg) == len(self.X_context) == len(self.y), "Data length mismatch!"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        ppg = torch.from_numpy(self.X_ppg[idx])     # (1, 64)
        context = torch.from_numpy(self.X_context[idx]) # (8,)
        target = torch.from_numpy(np.array([self.y[idx]])) # (1,)

        return ppg, context, target

# --- 使用示例 ---
if __name__ == "__main__":
    data_directory = './processed_cdn_data_v2'
    if not os.path.exists(data_directory):
        print(f"Error: Directory '{data_directory}' not found.")
    else:
        try:
            print("\nTesting CdnDatasetV2...")
            train_dataset = CdnDatasetV2(data_directory, split='train')
            test_dataset = CdnDatasetV2(data_directory, split='test')
            print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

            loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            if len(loader) > 0:
                ppg_batch, context_batch, target_batch = next(iter(loader))
                print("\nSample batch shapes:")
                print("  PPG:", ppg_batch.shape)     # Expect: [4, 1, 64]
                print("  Context:", context_batch.shape) # Expect: [4, 8]
                print("  Target:", target_batch.shape)  # Expect: [4, 1]
            else:
                 print("DataLoader is empty.")
        except Exception as e:
            print(f"Error during testing: {e}")