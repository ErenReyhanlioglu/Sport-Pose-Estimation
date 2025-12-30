import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def train_model(model, train_loader, val_loader, config, device):
    global_cfg = config['training']
    cls_cfg = config.get('classifier', {})
    
    ls_val = cls_cfg.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls_val) 
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=global_cfg['learning_rate'],
        weight_decay=global_cfg.get('weight_decay', 0.0)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_f1 = 0.0
    early_stop_count = 0
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_prec': [], 'train_rec': [],
        'val_loss': [],   'val_acc': [],   'val_f1': [],   'val_prec': [],   'val_rec': []
    }

    print(f"\n{'CLASSIFIER TRAINING START':^110}")
    print(f"Params: Epochs={global_cfg['epochs']}, BS={global_cfg['batch_size']}, LR={global_cfg['learning_rate']}, WD={global_cfg.get('weight_decay', 0.0)}")
    print(f"Config: LabelSmoothing={ls_val}, EarlyStop={global_cfg['early_stopping']['enable']} (P={global_cfg['early_stopping']['patience']})")
    print("="*110)
    
    header = f"{'Ep':^3} | {'TrLoss':^7} {'TrAcc':^6} {'TrF1':^6} {'TrP':^6} {'TrR':^6} | {'ValLoss':^7} {'ValAcc':^6} {'ValF1':^6} {'ValP':^6} {'ValR':^6}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for epoch in range(global_cfg['epochs']):
        model.train()
        train_loss = 0
        all_preds, all_targets = [], []
        
        for x_pose, x_sensor, y in tqdm(train_loader, desc=f"CLS Epoch {epoch+1}", leave=False):
            x_pose, x_sensor, y = x_pose.to(device), x_sensor.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_pose, x_sensor)
            loss = criterion(outputs, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        t_loss = train_loss / len(train_loader)
        t_acc = (np.array(all_preds) == np.array(all_targets)).mean()
        t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)

        v_loss, v_acc, v_prec, v_rec, v_f1 = validate_classifier(model, val_loader, criterion, device)
        
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc);   history['val_acc'].append(v_acc)
        history['train_f1'].append(t_f1);     history['val_f1'].append(v_f1)
        history['train_prec'].append(t_prec); history['val_prec'].append(v_prec)
        history['train_rec'].append(t_rec);   history['val_rec'].append(v_rec)
        
        print(f"{epoch+1:^3} | {t_loss:^7.4f} {t_acc:^6.4f} {t_f1:^6.4f} {t_prec:^6.4f} {t_rec:^6.4f} | "
              f"{v_loss:^7.4f} {v_acc:^6.4f} {v_f1:^6.4f} {v_prec:^6.4f} {v_rec:^6.4f}")
        
        scheduler.step(v_f1)
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            early_stop_count = 0
            best_model_state = model.state_dict()
        else:
            early_stop_count += 1
            if global_cfg['early_stopping']['enable'] and early_stop_count >= global_cfg['early_stopping']['patience']:
                print(f"--- Early stopping! (Best Val F1: {best_f1:.4f}) ---")
                break
    
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
        
    return model, history

def validate_classifier(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x_pose, x_sensor, y in loader:
            x_pose, x_sensor, y = x_pose.to(device), x_sensor.to(device), y.to(device)
            outputs = model(x_pose, x_sensor)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    v_loss = val_loss / len(loader)
    v_acc = (np.array(all_preds) == np.array(all_targets)).mean()
    v_prec, v_rec, v_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    return v_loss, v_acc, v_prec, v_rec, v_f1

def train_autoencoder(model, train_loader, val_loader, config, device):
    global_cfg = config['training']
    
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss() 
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=global_cfg['learning_rate'],
        weight_decay=global_cfg.get('weight_decay', 0.0)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    early_stop_count = 0
    
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}
    
    print(f"\n{'AUTOENCODER TRAINING START':^85}")
    print(f"Params: Epochs={global_cfg['epochs']}, BS={global_cfg['batch_size']}, LR={global_cfg['learning_rate']}, WD={global_cfg.get('weight_decay', 0.0)}")
    print(f"Config: EarlyStop={global_cfg['early_stopping']['enable']} (P={global_cfg['early_stopping']['patience']})")
    print("="*85)

    for epoch in range(global_cfg['epochs']):
        model.train()
        train_loss, train_mae = 0.0, 0.0
        
        for x_pose, _, _ in tqdm(train_loader, desc=f"AE Epoch {epoch+1}", leave=False):
            x_pose = x_pose.to(device)
            optimizer.zero_grad()
            reconstructed = model(x_pose)
            
            loss = criterion(reconstructed, x_pose)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            with torch.no_grad():
                train_mae += mae_criterion(reconstructed, x_pose).item()

        t_loss = train_loss / len(train_loader)
        t_mae = train_mae / len(train_loader)

        v_loss, v_mae = validate_autoencoder(model, val_loader, criterion, mae_criterion, device)
        
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_mae'].append(t_mae);   history['val_mae'].append(v_mae)
        
        print(f"{epoch+1:^3} | MSE: {t_loss:.4f} MAE: {t_mae:.4f} | Val MSE: {v_loss:.4f} Val MAE: {v_mae:.4f}")
        
        scheduler.step(v_loss)
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            early_stop_count = 0
            best_model_state = model.state_dict()
        else:
            early_stop_count += 1
            if global_cfg['early_stopping']['enable'] and early_stop_count >= global_cfg['early_stopping']['patience']:
                print(f"--- Early stopping! (Best Val Loss: {best_val_loss:.6f}) ---")
                break

    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
        
    return model, history

def validate_autoencoder(model, loader, criterion, mae_criterion, device):
    model.eval()
    val_loss, val_mae = 0.0, 0.0
    
    with torch.no_grad():
        for x_pose, _, _ in loader:
            x_pose = x_pose.to(device)
            reconstructed = model(x_pose)
            val_loss += criterion(reconstructed, x_pose).item()
            val_mae += mae_criterion(reconstructed, x_pose).item()
            
    return val_loss / len(loader), val_mae / len(loader)

def train_multitask(model, train_loader, val_loader, config, device):
    global_cfg = config['training']
    mtl_cfg = config['multitask']
    
    ls_val = mtl_cfg.get('label_smoothing', 0.0)
    use_uncertainty = mtl_cfg.get('uncertainty_weighting', {}).get('enable', False)
    
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=ls_val)
    criterion_recon = nn.MSELoss()
    mae_criterion = nn.L1Loss() 
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=global_cfg['learning_rate'],
        weight_decay=global_cfg.get('weight_decay', 0.0)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_f1 = 0.0
    early_stop_count = 0
    
    history = {
        'train_loss': [], 'train_cls_loss': [], 'train_recon_loss': [],
        'train_f1': [],   'train_acc': [],      'train_prec': [],       'train_rec': [],
        'train_mae': [],
        
        'val_loss': [],   'val_cls_loss': [],   'val_recon_loss': [],
        'val_f1': [],     'val_acc': [],        'val_prec': [],         'val_rec': [],
        'val_mae': []
    }

    print(f"\n{'MULTI-TASK TRAINING START (CLS + RECON)':^140}")
    print(f"Params: Epochs={global_cfg['epochs']}, BS={global_cfg['batch_size']}, LR={global_cfg['learning_rate']}, WD={global_cfg.get('weight_decay', 0.0)}")
    print(f"Config: Uncertainty={use_uncertainty}, LabelSmoothing={ls_val}, GradNoise={mtl_cfg.get('gradient_noise', 0.0)}")
    print(f"Config: EarlyStop={global_cfg['early_stopping']['enable']} (P={global_cfg['early_stopping']['patience']})")
    print("="*140)
    
    header = (
        f"{'Ep':^3} | "
        f"{'TrTot':^7} {'TrCls':^7} {'TrRec':^7} | {'TrAcc':^5} {'TrF1':^5} {'TrMAE':^6} | "
        f"{'ValTot':^7} {'ValCls':^7} {'ValRec':^7} | {'ValAcc':^5} {'ValF1':^5} {'ValMAE':^6}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for epoch in range(global_cfg['epochs']):
        model.train()
        
        total_loss_sum = 0
        cls_loss_sum = 0
        recon_loss_sum = 0
        recon_mae_sum = 0 
        
        all_preds, all_targets = [], []
        
        for x_pose, x_sensor, y in tqdm(train_loader, desc=f"MTL Epoch {epoch+1}", leave=False):
            x_pose, x_sensor, y = x_pose.to(device), x_sensor.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            cls_logits, recon_pose = model(x_pose, x_sensor)
            
            loss_cls = criterion_cls(cls_logits, y)
            loss_recon = criterion_recon(recon_pose, x_pose)
            
            with torch.no_grad():
                mae = mae_criterion(recon_pose, x_pose)
                recon_mae_sum += mae.item()
            
            if use_uncertainty:
                precision1 = torch.exp(-model.log_vars[0])
                loss1 = precision1 * loss_cls + model.log_vars[0]
                precision2 = torch.exp(-model.log_vars[1])
                loss2 = precision2 * loss_recon + model.log_vars[1]
                loss_total = loss1 + loss2
            else:
                w_cls = mtl_cfg['loss_weights']['classifier']
                w_recon = mtl_cfg['loss_weights']['reconstruction']
                loss_total = (w_cls * loss_cls) + (w_recon * loss_recon)
            
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss_sum += loss_total.item()
            cls_loss_sum += loss_cls.item()
            recon_loss_sum += loss_recon.item()
            
            all_preds.extend(cls_logits.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        n_batches = len(train_loader)
        t_loss = total_loss_sum / n_batches
        t_loss_cls = cls_loss_sum / n_batches
        t_loss_recon = recon_loss_sum / n_batches
        t_mae = recon_mae_sum / n_batches
        
        t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
        t_acc = (np.array(all_preds) == np.array(all_targets)).mean()

        v_metrics = validate_multitask(model, val_loader, criterion_cls, criterion_recon, mae_criterion,
                                       mtl_cfg['loss_weights']['classifier'], mtl_cfg['loss_weights']['reconstruction'], device)
        v_f1 = v_metrics['f1']
        
        history['train_loss'].append(t_loss); history['val_loss'].append(v_metrics['total_loss'])
        history['train_cls_loss'].append(t_loss_cls); history['val_cls_loss'].append(v_metrics['cls_loss'])
        history['train_recon_loss'].append(t_loss_recon); history['val_recon_loss'].append(v_metrics['recon_loss'])
        
        history['train_acc'].append(t_acc); history['val_acc'].append(v_metrics['acc'])
        history['train_f1'].append(t_f1); history['val_f1'].append(v_f1)
        history['train_prec'].append(t_prec); history['val_prec'].append(v_metrics['prec'])
        history['train_rec'].append(t_rec); history['val_rec'].append(v_metrics['rec'])
        
        history['train_mae'].append(t_mae); history['val_mae'].append(v_metrics['mae'])
        
        print(f"{epoch+1:^3} | "
              f"{t_loss:^7.4f} {t_loss_cls:^7.4f} {t_loss_recon:^7.4f} | {t_acc:^5.4f} {t_f1:^5.4f} {t_mae:^6.4f} | "
              f"{v_metrics['total_loss']:^7.4f} {v_metrics['cls_loss']:^7.4f} {v_metrics['recon_loss']:^7.4f} | {v_metrics['acc']:^5.4f} {v_f1:^5.4f} {v_metrics['mae']:^6.4f}")
        
        scheduler.step(v_f1)
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            early_stop_count = 0
            best_model_state = model.state_dict()
        else:
            early_stop_count += 1
            if global_cfg['early_stopping']['enable'] and early_stop_count >= global_cfg['early_stopping']['patience']:
                print(f"--- Early stopping! (Best Val F1: {best_f1:.4f}) ---")
                break
                
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
        
    return model, history

def validate_multitask(model, loader, crit_cls, crit_recon, crit_mae, w_cls, w_recon, device):
    model.eval()
    total_loss, cls_loss, recon_loss, mae_sum = 0, 0, 0, 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x_pose, x_sensor, y in loader:
            x_pose, x_sensor, y = x_pose.to(device), x_sensor.to(device), y.to(device)
            
            c_out, r_out = model(x_pose, x_sensor)
            
            l_c = crit_cls(c_out, y)
            l_r = crit_recon(r_out, x_pose)
            l_t = (w_cls * l_c) + (w_recon * l_r)
            mae = crit_mae(r_out, x_pose)
            
            total_loss += l_t.item()
            cls_loss += l_c.item()
            recon_loss += l_r.item()
            mae_sum += mae.item()
            
            all_preds.extend(c_out.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    n = len(loader)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    acc = (np.array(all_preds) == np.array(all_targets)).mean()
    
    return {
        'total_loss': total_loss / n,
        'cls_loss': cls_loss / n,
        'recon_loss': recon_loss / n,
        'mae': mae_sum / n,
        'f1': f1,
        'acc': acc,
        'prec': prec,
        'rec': rec
    }