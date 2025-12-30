import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MMFitDataset(Dataset):
    def __init__(self, pose, sensor, y, augment=False, noise_std=0.05):
        self.pose = pose
        self.sensor = sensor
        self.y = y
        self.augment = augment 
        self.noise_std = noise_std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        pose_sample = self.pose[idx]
        sensor_sample = self.sensor[idx]
        label = self.y[idx]
        
        if self.augment and self.noise_std > 0:
            noise_p = torch.randn_like(pose_sample) * self.noise_std
            pose_sample = pose_sample + noise_p
            
            noise_s = torch.randn_like(sensor_sample) * (self.noise_std * 0.6)
            sensor_sample = sensor_sample + noise_s
            
        return pose_sample, sensor_sample, label

def load_all_sessions(data_dir):
    pose_list, sensor_list, y_list = [], [], []
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
    print(f"[Veri Yükleme] {len(all_files)} dosya okunuyor...")
    
    for filename in all_files:
        file_path = os.path.join(data_dir, filename)
        try:
            data = torch.load(file_path)
            pose_list.append(data['pose'])
            sensor_list.append(data['sensor'])
            y_list.append(data['y'])
        except Exception as e:
            print(f"   [HATA] {filename}: {e}")
            
    if not pose_list: raise RuntimeError("Veri bulunamadı!")
    return torch.cat(pose_list), torch.cat(sensor_list), torch.cat(y_list)

def get_dataloaders(config):
    data_dir = config['data']['dataset_path']
    batch_size = config['training']['batch_size']
    
    noise_val = 0.05
    
    if 'multitask' in config and 'gradient_noise' in config['multitask']:
        noise_val = config['multitask']['gradient_noise']
    elif 'training' in config and 'gradient_noise' in config['training']:
        noise_val = config['training']['gradient_noise']
    
    if noise_val <= 0:
        noise_val = 0.0 

    print("="*60)
    print(f"[Preprocessing] STRATIFIED SPLIT + NOISE INJECTION (Std: {noise_val})")
    print("="*60)
    
    all_pose, all_sensor, all_y = load_all_sessions(data_dir)
    
    mean_p = all_pose.mean(dim=(0, 1), keepdim=True)
    std_p = all_pose.std(dim=(0, 1), keepdim=True) + 1e-6
    all_pose = (all_pose - mean_p) / std_p
    
    mean_s = all_sensor.mean(dim=(0, 1), keepdim=True)
    std_s = all_sensor.std(dim=(0, 1), keepdim=True) + 1e-6
    all_sensor = (all_sensor - mean_s) / std_s
    
    X_p_train, X_p_temp, X_s_train, X_s_temp, y_train, y_temp = train_test_split(
        all_pose.numpy(), all_sensor.numpy(), all_y.numpy(), 
        test_size=0.30, random_state=42, stratify=all_y.numpy()
    )
    
    X_p_val, X_p_test, X_s_val, X_s_test, y_val, y_test = train_test_split(
        X_p_temp, X_s_temp, y_temp, 
        test_size=0.50, random_state=42, stratify=y_temp
    )
    
    train_data = (torch.tensor(X_p_train), torch.tensor(X_s_train), torch.tensor(y_train))
    val_data   = (torch.tensor(X_p_val),   torch.tensor(X_s_val),   torch.tensor(y_val))
    test_data  = (torch.tensor(X_p_test),  torch.tensor(X_s_test),  torch.tensor(y_test))
    
    train_loader = DataLoader(
        MMFitDataset(*train_data, augment=True, noise_std=noise_val), 
        batch_size=batch_size, shuffle=True
    )
    
    val_loader   = DataLoader(
        MMFitDataset(*val_data, augment=False), 
        batch_size=batch_size, shuffle=False
    )
    
    test_loader  = DataLoader(
        MMFitDataset(*test_data, augment=False), 
        batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader