import os
import torch
import yaml
import json
import numpy as np
from scripts import experiment, preprocessing, model, trainer, evaluater, plots

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_logs(history, experiment_path):
    log_path = os.path.join(experiment_path, 'logs', 'training_history.json')
    
    serializable_history = {}
    
    for k, vals in history.items():
        serializable_history[k] = [
            float(v) if isinstance(v, (np.float32, np.float64, np.ndarray, torch.Tensor)) 
            else v 
            for v in vals
        ]
        
    with open(log_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)
        
    print(f"\n[Main] Loglar kaydedildi: {log_path}")
    print(f"[Main] Kaydedilen Metrikler: {list(serializable_history.keys())}")

def run_pipeline(config_path, mode='cls'):
    valid_modes = ['ae', 'cls', 'mtl']
    if mode not in valid_modes:
        raise ValueError(f"Hatalı mod: {mode}. Beklenenler: {valid_modes}")

    config = load_config(config_path)
    train_cfg = config['training'] 
    
    set_seeds(train_cfg['random_state'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{' PIPELINE STARTED ':^60}")
    print(f"{' Mode: ' + mode.upper() + ' ':^60}")
    print("="*60)

    print("\n--- [1] Data Loading ---")
    train_loader, val_loader, test_loader = preprocessing.get_dataloaders(config)

    if mode == 'ae':
        print(f"\n--- [2] Setup: Autoencoder ---")
        experiment_path = experiment.setup_experiment(config, task_name="AE")
        models_dir = os.path.join(experiment_path, 'models')
        
        ae_model = model.PoseAutoencoder(
            seq_len=config['model']['seq_len'],
            n_features=config['model']['pose_input_size'],
            embedding_dim=config['autoencoder']['embedding_dim']
        ).to(device)
        
        trained_ae, ae_history = trainer.train_autoencoder(
            ae_model, train_loader, val_loader, config, device
        )
        
        torch.save(trained_ae.state_dict(), os.path.join(models_dir, 'best_autoencoder.pth'))
        
        save_logs(ae_history, experiment_path)
        plots.save_autoencoder_training_plots(ae_history, experiment_path)
        
        print("\n--- [3] Evaluation ---")
        metrics, _, _, _, reconstruction_errors = evaluater.evaluate_model(
            model=trained_ae,
            test_loader=test_loader,
            config=config,
            experiment_path=experiment_path,
            device=device,
            mode='ae'
        )
        
        if reconstruction_errors is not None:
            plots.save_reconstruction_error_distribution(reconstruction_errors, experiment_path)
        
        print(f"DONE. Output: {experiment_path}")

    elif mode == 'cls':
        print(f"\n--- [2] Setup: Classifier ---")
        experiment_path = experiment.setup_experiment(config, task_name="CLS")
        models_dir = os.path.join(experiment_path, 'models')
        
        print(f"Config: LR={train_cfg['learning_rate']}, "  
              f"BS={train_cfg['batch_size']}, "            
              f"LS={config.get('classifier', {}).get('label_smoothing', 0.0)}")
        
        cls_model = model.MultiModalClassifier(
            pose_dim=config['model']['pose_input_size'],
            sensor_dim=config['model']['sensor_input_size'],
            hidden_size=config['model']['hidden_size'],
            num_classes=config['model']['num_classes']
        ).to(device)

        trained_cls, history = trainer.train_model(
            model=cls_model,
            train_loader=train_loader,
            val_loader=val_loader, 
            config=config,
            device=device
        )
        
        torch.save(trained_cls.state_dict(), os.path.join(models_dir, 'best_classifier.pth'))
        
        save_logs(history, experiment_path)

        print("\n--- [3] Evaluation ---")
        metrics, report, y_true, y_pred, _ = evaluater.evaluate_model(
            model=trained_cls,
            test_loader=test_loader,
            config=config,
            experiment_path=experiment_path,
            device=device,
            mode='cls'
        )

        plots.save_training_history_plots(history, experiment_path)
        plots.save_comprehensive_evaluation_plots(y_true, y_pred, config, experiment_path)
        plots.save_confusion_matrix_plot(y_true, y_pred, config, experiment_path)
        
        print(f"DONE. Output: {experiment_path}")

    elif mode == 'mtl':
        print(f"\n--- [2] Setup: Multi-Task Learning ---")
        experiment_path = experiment.setup_experiment(config, task_name="MTL")
        models_dir = os.path.join(experiment_path, 'models')
        
        mtl_cfg = config['multitask']
        use_uncertainty = mtl_cfg.get('uncertainty_weighting', {}).get('enable', False)
        ablation_cfg = config.get('ablation', {})
        
        print(f"\n[Ablation Study Configuration]")
        print(f" > Fusion Type  : {ablation_cfg.get('fusion_type', 'concat')}")
        print(f" > RNN Type     : {ablation_cfg.get('rnn_type', 'lstm')}")
        print(f" > Decoder Type : {ablation_cfg.get('decoder_type', 'simple')}")
        print(f" > Uncertainty  : {use_uncertainty}")
        print("-" * 40)
        
        print(f"Config: LR={train_cfg['learning_rate']}") 
        print(f"        LabelSmoothing={mtl_cfg.get('label_smoothing', 0.0)}")
        print(f"        GradNoise={mtl_cfg.get('gradient_noise', 0.0)}")
        if not use_uncertainty:
            print(f"        Weights={mtl_cfg['loss_weights']}")
        
        mtl_model = model.MultiTaskNetwork(
            pose_dim=config['model']['pose_input_size'],
            sensor_dim=config['model']['sensor_input_size'],
            hidden_size=config['model']['hidden_size'],
            num_classes=config['model']['num_classes'],
            uncertainty_weighting=use_uncertainty,
            ablation_config=ablation_cfg
        ).to(device)
        
        trained_mtl, history = trainer.train_multitask(
            model=mtl_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
        torch.save(trained_mtl.state_dict(), os.path.join(models_dir, 'best_multitask_network.pth'))
        
        save_logs(history, experiment_path)
        
        print("\n--- [3] Evaluation (Dual Task) ---")
        metrics, report, y_true, y_pred, reconstruction_errors = evaluater.evaluate_model(
            model=trained_mtl,
            test_loader=test_loader,
            config=config,
            experiment_path=experiment_path,
            device=device,
            mode='mtl'
        )
        
        plots.save_training_history_plots(history, experiment_path)
        plots.save_comprehensive_evaluation_plots(y_true, y_pred, config, experiment_path)
        plots.save_confusion_matrix_plot(y_true, y_pred, config, experiment_path)
        plots.save_sub_loss_comparison_plot(history, experiment_path)
        
        if reconstruction_errors is not None:
            plots.save_reconstruction_error_distribution(reconstruction_errors, experiment_path)
            
        print(f"DONE. Output: {experiment_path}")