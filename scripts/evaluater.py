import torch
import torch.nn as nn
import numpy as np
import os
import json
from sklearn.metrics import classification_report

def evaluate_model(model, test_loader, config, experiment_path, device, mode='cls'):
   
    model.eval()
    print(f"\n[Evaluater] Değerlendirme Modu: {mode.upper()}")

    if mode == 'cls':
        return _evaluate_classifier(model, test_loader, config, experiment_path, device)
    elif mode == 'ae':
        return _evaluate_autoencoder(model, test_loader, config, experiment_path, device)
    elif mode == 'mtl':
        return _evaluate_multitask(model, test_loader, config, experiment_path, device)
    else:
        raise ValueError(f"Geçersiz mod: {mode}")

def _evaluate_classifier(model, test_loader, config, experiment_path, device):
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for x_pose, x_sensor, y in test_loader:
            x_pose, x_sensor, y = x_pose.to(device), x_sensor.to(device), y.to(device)
            
            cls_logits = model(x_pose, x_sensor)
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(cls_logits.argmax(1).cpu().numpy())

    class_labels = list(config['evaluation']['class_labels'].keys())
    
    report_str = classification_report(y_true, y_pred, target_names=class_labels, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0)
    
    metrics = {
        'test_accuracy': report_dict['accuracy'],
        'macro_f1': report_dict['macro avg']['f1-score'],
        'weighted_f1': report_dict['weighted avg']['f1-score']
    }
    
    _save_results(experiment_path, metrics, report_str)
    
    print(f"\n[Evaluater] CLS Değerlendirme Tamamlandı.")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Macro F1:      {metrics['macro_f1']:.4f}")
    
    return metrics, report_dict, np.array(y_true), np.array(y_pred), None

def _evaluate_autoencoder(model, test_loader, config, experiment_path, device):
    mse_criterion = nn.MSELoss(reduction='none') 
    mae_criterion = nn.L1Loss(reduction='none')
    
    all_mse_losses = []
    all_mae_losses = []
    
    print("Reconstruction hataları hesaplanıyor...")
    
    with torch.no_grad():
        for x_pose, _, _ in test_loader:
            x_pose = x_pose.to(device)
            reconstructed = model(x_pose)
            
            mse_loss = mse_criterion(reconstructed, x_pose).mean(dim=(1, 2)) 
            mae_loss = mae_criterion(reconstructed, x_pose).mean(dim=(1, 2))
            
            all_mse_losses.extend(mse_loss.cpu().numpy())
            all_mae_losses.extend(mae_loss.cpu().numpy())
    
    all_mse_losses = np.array(all_mse_losses)
    
    metrics = {
        'recon_mean_mse': float(np.mean(all_mse_losses)),
        'recon_std_mse': float(np.std(all_mse_losses)),
        'recon_mean_mae': float(np.mean(all_mae_losses))
    }
    
    with open(os.path.join(experiment_path, 'evaluation_results.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    np.save(os.path.join(experiment_path, 'reconstruction_errors.npy'), all_mse_losses)
    
    print(f"\n[Evaluater] AE Değerlendirme Tamamlandı.")
    print(f"Mean MSE: {metrics['recon_mean_mse']:.6f} ± {metrics['recon_std_mse']:.6f}")

    return metrics, None, None, None, all_mse_losses

def _evaluate_multitask(model, test_loader, config, experiment_path, device):
  
    mse_criterion = nn.MSELoss(reduction='none')
    mae_criterion = nn.L1Loss(reduction='none')

    y_true, y_pred = [], []
    all_mse_losses = []
    all_mae_losses = []

    print("Multi-Task Performansı Hesaplanıyor...")

    with torch.no_grad():
        for x_pose, x_sensor, y in test_loader:
            x_pose, x_sensor, y = x_pose.to(device), x_sensor.to(device), y.to(device)

            cls_logits, recon_pose = model(x_pose, x_sensor)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(cls_logits.argmax(1).cpu().numpy())

            mse_loss = mse_criterion(recon_pose, x_pose).mean(dim=(1, 2))
            mae_loss = mae_criterion(recon_pose, x_pose).mean(dim=(1, 2))
            
            all_mse_losses.extend(mse_loss.cpu().numpy())
            all_mae_losses.extend(mae_loss.cpu().numpy())

    class_labels = list(config['evaluation']['class_labels'].keys())
    
    report_str = classification_report(y_true, y_pred, target_names=class_labels, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0)
    
    all_mse_losses = np.array(all_mse_losses)
    all_mae_losses = np.array(all_mae_losses)

    metrics = {
        'cls_accuracy': report_dict['accuracy'],
        'cls_macro_f1': report_dict['macro avg']['f1-score'],
        'cls_weighted_f1': report_dict['weighted avg']['f1-score'],
        
        'recon_mean_mse': float(np.mean(all_mse_losses)),
        'recon_std_mse': float(np.std(all_mse_losses)),
        'recon_min_mse': float(np.min(all_mse_losses)),
        'recon_max_mse': float(np.max(all_mse_losses)),
        'recon_mean_mae': float(np.mean(all_mae_losses))
    }

    _save_results(experiment_path, metrics, report_str)
    np.save(os.path.join(experiment_path, 'reconstruction_errors.npy'), all_mse_losses)

    print("\n" + "="*70)
    print(f"{'MULTI-TASK (MTL) GENEL PERFORMANSI':^70}")
    print("="*70)
    print(f"Task 1 (Sınıflandırma) -> Accuracy: {metrics['cls_accuracy']:.4f} | F1: {metrics['cls_macro_f1']:.4f}")
    print(f"Task 2 (Düzeltme/Recon)-> Mean MSE: {metrics['recon_mean_mse']:.6f}")
    print("="*70)
    print(report_str)

    return metrics, report_dict, np.array(y_true), np.array(y_pred), all_mse_losses

def _save_results(experiment_path, metrics, report_str):
   
    with open(os.path.join(experiment_path, 'evaluation_results.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(experiment_path, 'detailed_class_report.txt'), 'w') as f:
        f.write(report_str)