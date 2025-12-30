import torch
import numpy as np
import torch.nn.functional as F

def generate_mtl_report(mtl_model, test_loader, feature_names, device, stats, label_map):
    """
    MTL Modeli için Sınıflandırma ve Hareket Düzeltme Raporu Oluşturur.
    """
    mtl_model.eval()
    pose_mean, pose_std = stats
    
    p_mean = pose_mean.cpu().numpy().squeeze()
    p_std = pose_std.cpu().numpy().squeeze()

    iterator = iter(test_loader)
    
    try:
        batch_data = next(iterator)
        if len(batch_data) == 3:
            x_pose, x_sensor, y_label = batch_data
        else:
            print("Hata: Dataloader 3 bileşen (Pose, Sensor, Label) döndürmüyor.")
            return
    except StopIteration:
        return

    sample_pose = x_pose[0].unsqueeze(0).to(device)     # (1, 90, 61)
    sample_sensor = x_sensor[0].unsqueeze(0).to(device) # (1, 90, 6)
    true_class_id = y_label[0].item()

    with torch.no_grad():
        class_logits, recon_pose = mtl_model(sample_pose, sample_sensor)

    probs = F.softmax(class_logits, dim=1)
    pred_class_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class_id].item() * 100

    true_label_str = label_map.get(true_class_id, "Unknown").upper()
    pred_label_str = label_map.get(pred_class_id, "Unknown").upper()
    
    match_icon = "T" if pred_class_id == true_class_id else "F"

    print(f"\n{'='*60}")
    print(f"  MTL MODEL ANALİZ RAPORU (V2.3 Entegrasyonu)")
    print(f"{'='*60}")
    
    print(f"\n[SINIFLANDIRMA SONUCU]")
    print(f"Gerçek Sınıf   : {true_label_str} (ID: {true_class_id})")
    print(f"Tahmin Edilen  : {pred_label_str} (ID: {pred_class_id}) {match_icon}")
    print(f"Güven Skoru    : %{confidence:.2f}")
    
    user_norm = torch.mean(sample_pose, dim=1).squeeze(0).cpu().numpy()
    ideal_norm = torch.mean(recon_pose, dim=1).squeeze(0).cpu().numpy()
    
    start_idx = 51 

    print(f"\n[DETAYLI EKLEM ANALİZİ]")
    print(f"{'EKLEM':<20} | {'DURUM':<10} | {'MEVCUT':<8} | {'İDEAL':<8} | {'DÜZELTME ÖNERİSİ'}")
    print("-" * 100)

    for i, name in enumerate(feature_names):
        idx = start_idx + i
        if idx >= len(user_norm): break

        user_deg = ((user_norm[idx] * p_std[idx]) + p_mean[idx]) * 180.0
        ideal_deg = ((ideal_norm[idx] * p_std[idx]) + p_mean[idx]) * 180.0
        
        diff = ideal_deg - user_deg
        norm_error = abs(user_norm[idx] - ideal_norm[idx])
        
        threshold = 0.25 

        if norm_error < threshold:
            status = "DOĞRU"
            correction = "Koruyun."
        else:
            status = "HATALI"
            if diff > 0:
                correction = f"ARTIR (+{abs(diff):.1f}°)"
            else:
                correction = f"AZALT (-{abs(diff):.1f}°)"

        print(f"{name:<20} | {status:<10} | {user_deg:>6.1f}° | {ideal_deg:>6.1f}° | {correction}")
    
    print(f"{'='*60}\n")