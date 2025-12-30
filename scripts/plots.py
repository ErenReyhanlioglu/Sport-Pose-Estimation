import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE_TRAIN_VAL = ["#2ecc71", "#e74c3c"] 
PALETTE_MTL = ["#3498db", "#e67e22"]       

def save_training_history_plots(history, experiment_path):
    plots_dir = os.path.join(experiment_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    metrics_map = [
        ('Loss', 'train_loss', 'val_loss'),
        ('Accuracy', 'train_acc', 'val_acc'),
        ('F1-Score', 'train_f1', 'val_f1'),
        ('MAE (Reconstruction)', 'train_mae', 'val_mae'), 
        ('Precision', 'train_prec', 'val_prec'),
        ('Recall', 'train_rec', 'val_rec')
    ]
    
    active_metrics = [m for m in metrics_map if m[1] in history and len(history[m[1]]) > 0]
    
    n_plots = len(active_metrics)
    n_cols = 2
    n_rows = (n_plots + 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, (title, train_key, val_key) in enumerate(active_metrics):
        plt.subplot(n_rows, n_cols, i + 1)
        
        sns.lineplot(x=epochs, y=history[train_key], label='Train', 
                     color=PALETTE_TRAIN_VAL[0], linewidth=2, marker='o', markersize=4)
        sns.lineplot(x=epochs, y=history[val_key], label='Validation', 
                     color=PALETTE_TRAIN_VAL[1], linewidth=2, marker='o', markersize=4)
        
        plt.title(f'{title} Evolution', fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_metrics_summary.png'), dpi=300)
    plt.close()
    
    if 'val_cls_loss' in history and 'val_recon_loss' in history:
        save_mtl_loss_dynamics(history, plots_dir)

def save_mtl_loss_dynamics(history, save_dir):
    epochs = range(1, len(history['val_loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = PALETTE_MTL[0]
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Classifier Loss', color=color, fontweight='bold')
    ax1.plot(epochs, history['val_cls_loss'], color=color, linewidth=2, label='CLS Loss (Val)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(False) 

    ax2 = ax1.twinx()  
    color = PALETTE_MTL[1]
    ax2.set_ylabel('Reconstruction Loss (MSE)', color=color, fontweight='bold')
    ax2.plot(epochs, history['val_recon_loss'], color=color, linewidth=2, linestyle='--', label='Recon Loss (Val)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Multi-Task Learning Dynamics: Task Loss Balance', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mtl_loss_dynamics.png'), dpi=300)
    plt.close()

def save_comprehensive_evaluation_plots(y_true, y_pred, config, experiment_path):
    class_names = list(config['evaluation']['class_labels'].keys())
    plots_dir = os.path.join(experiment_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(class_names)))
    
    data = []
    for i, class_name in enumerate(class_names):
        data.append({'Class': class_name, 'Metric': 'Precision', 'Score': prec[i]})
        data.append({'Class': class_name, 'Metric': 'Recall',    'Score': rec[i]})
        data.append({'Class': class_name, 'Metric': 'F1-Score',  'Score': f1[i]})
    
    df_metrics = pd.DataFrame(data)
    
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_metrics, x='Class', y='Score', hue='Metric', palette='viridis')
    
    plt.title('Per-Class Performance Breakdown', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'per_class_metrics.png'), dpi=300)
    plt.close()

def save_confusion_matrix_plot(y_true, y_pred, config, experiment_path):
    class_names = list(config['evaluation']['class_labels'].keys())
    plots_dir = os.path.join(experiment_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    def plot_cm(cm_data, title, filename, fmt, cmap):
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_data, annot=True, fmt=fmt, cmap=cmap, 
                    xticklabels=class_names, yticklabels=class_names,
                    square=True, cbar_kws={"shrink": .8})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontweight='bold')
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=300)
        plt.close()

    plot_cm(cm, 'Confusion Matrix (Raw Counts)', 'cm_raw.png', 'd', 'Blues')
    
    cm_recall = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    plot_cm(cm_recall, 'Recall Matrix (Row Normalized)', 'cm_recall.png', '.2f', 'Greens')
    
    cm_prec = cm.astype('float') / (cm.sum(axis=0)[np.newaxis, :] + 1e-6)
    plot_cm(cm_prec, 'Precision Matrix (Column Normalized)', 'cm_precision.png', '.2f', 'Purples')

def save_autoencoder_training_plots(history, experiment_path):
    save_training_history_plots(history, experiment_path)

def save_reconstruction_error_distribution(errors, experiment_path):
    plots_dir = os.path.join(experiment_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True, color='darkcyan', edgecolor='black', alpha=0.6)
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    percentile_95 = np.percentile(errors, 95)
    
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.4f}')
    plt.axvline(percentile_95, color='orange', linestyle=':', linewidth=2, label=f'95th Percentile: {percentile_95:.4f}')
    
    plt.title('Reconstruction Error Distribution (Quality Analysis)', fontsize=14)
    plt.xlabel('Mean Squared Error (MSE)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'reconstruction_dist_hist.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=errors, color='lightblue', flierprops={"marker": "x", "markerfacecolor": "red"})
    plt.title('Reconstruction Error Boxplot (Outlier Detection)', fontsize=14)
    plt.xlabel('MSE Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'reconstruction_dist_boxplot.png'), dpi=300)
    plt.close()

def save_sub_loss_comparison_plot(history, experiment_path):
    plots_dir = os.path.join(experiment_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    if 'train_cls_loss' not in history or 'val_cls_loss' not in history:
        print("[Plot] Sub-loss verisi bulunamadı, grafik atlanıyor.")
        return

    epochs = range(1, len(history['train_cls_loss']) + 1)
    
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_cls_loss'], label='Train Cls Loss', color='#2980b9', linewidth=2)
    plt.plot(epochs, history['val_cls_loss'], label='Val Cls Loss', color='#3498db', linestyle='--', linewidth=2)
    
    plt.title('Task 1: Classification Loss Dynamics', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('CrossEntropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_recon_loss'], label='Train Recon Loss', color='#c0392b', linewidth=2)
    plt.plot(epochs, history['val_recon_loss'], label='Val Recon Loss', color='#e74c3c', linestyle='--', linewidth=2)
    
    plt.title('Task 2: Reconstruction Loss Dynamics', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'sub_loss_comparison.png'), dpi=300)
    plt.close()
    print(f"[Plot] Sub-loss grafiği kaydedildi: {os.path.join(plots_dir, 'sub_loss_comparison.png')}")