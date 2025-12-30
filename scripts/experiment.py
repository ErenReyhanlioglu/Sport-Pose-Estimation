import os
import yaml
from datetime import datetime

def setup_experiment(config, task_name="experiment"):
    
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") 
    experiment_name = f"experiment_{task_name}_{timestamp}"
    
    base_output_path = config['data']['output_path']
    os.makedirs(base_output_path, exist_ok=True)
    
    experiment_path = os.path.join(base_output_path, experiment_name)
    
    sub_folders = ['plots', 'models', 'logs']
    for folder in sub_folders:
        os.makedirs(os.path.join(experiment_path, folder), exist_ok=True)
    
    snapshot_path = os.path.join(experiment_path, 'config_snapshot.yaml')
    with open(snapshot_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[Experiment] Deney klasörü oluşturuldu: {experiment_name}")
    print(f"[Experiment] Tam Yol: {experiment_path}")
    
    return experiment_path