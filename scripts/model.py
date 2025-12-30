import torch
import torch.nn as nn

class MultiModalClassifier(nn.Module):
    def __init__(self, pose_dim=61, sensor_dim=6, hidden_size=128, num_classes=10):
        super(MultiModalClassifier, self).__init__()
        
        self.pose_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), 
            nn.Dropout2d(0.4),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), 
            nn.Dropout2d(0.4)
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 90, pose_dim)
            out = self.pose_cnn(dummy)
            self.pose_lstm_in = out.shape[1] * out.shape[3]
            
        self.pose_lstm = nn.LSTM(self.pose_lstm_in, hidden_size, batch_first=True, bidirectional=True)
        
        self.sensor_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 1), padding=(1, 0)), 
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), 
            nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)) 
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 90, sensor_dim)
            out = self.sensor_cnn(dummy)
            self.sensor_lstm_in = out.shape[1] * out.shape[3]
            
        self.sensor_lstm = nn.LSTM(self.sensor_lstm_in, hidden_size, batch_first=True, bidirectional=True)
        
        fusion_dim = (hidden_size * 2) + (hidden_size * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_pose, x_sensor):
        p = x_pose.unsqueeze(1) 
        p = self.pose_cnn(p)    
        p = p.permute(0, 2, 1, 3) 
        batch, time, c, f = p.size()
        p = p.reshape(batch, time, c * f)
        
        self.pose_lstm.flatten_parameters()
        p_out, _ = self.pose_lstm(p)
        p_feat = torch.mean(p_out, dim=1) 
        
        s = x_sensor.unsqueeze(1) 
        s = self.sensor_cnn(s)
        s = s.permute(0, 2, 1, 3)
        batch, time, c, f = s.size()
        s = s.reshape(batch, time, c * f)
        
        self.sensor_lstm.flatten_parameters()
        s_out, _ = self.sensor_lstm(s)
        s_feat = torch.mean(s_out, dim=1) 
        
        combined = torch.cat([p_feat, s_feat], dim=1)
        logits = self.classifier(combined)
        
        return logits


class PoseAutoencoder(nn.Module):
    def __init__(self, seq_len=90, n_features=61, embedding_dim=32):
        super(PoseAutoencoder, self).__init__()
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.encoder_lstm = nn.LSTM(64, embedding_dim, batch_first=True)
        
        self.decoder_lstm = nn.LSTM(embedding_dim, 64, batch_first=True)
        self.decoder_fc = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, n_features)
        )
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

    def forward(self, x):
        batch_size = x.size(0)
        x_reshaped = x.reshape(-1, self.n_features)
        
        enc = self.encoder_fc(x_reshaped).reshape(batch_size, self.seq_len, -1)
        _, (hidden, _) = self.encoder_lstm(enc)
        embedding = hidden.squeeze(0)
        
        dec_in = embedding.unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.decoder_lstm(dec_in)
        lstm_out = lstm_out.reshape(-1, 64)
        recon = self.decoder_fc(lstm_out).reshape(batch_size, self.seq_len, self.n_features)
        
        return recon


class MultiTaskNetwork(nn.Module):
    def __init__(self, pose_dim=61, sensor_dim=6, hidden_size=128, num_classes=10, 
                 uncertainty_weighting=False, ablation_config=None):
        super(MultiTaskNetwork, self).__init__()
        
        if ablation_config is None:
            ablation_config = {'fusion_type': 'concat', 'rnn_type': 'lstm', 'decoder_type': 'simple'}
            
        self.fusion_type = ablation_config.get('fusion_type', 'concat')
        self.rnn_type = ablation_config.get('rnn_type', 'lstm')
        self.decoder_type = ablation_config.get('decoder_type', 'simple')
        self.uncertainty_weighting = uncertainty_weighting

        print(f"\n[Model Init] Ablation Settings: Fusion='{self.fusion_type}', RNN='{self.rnn_type}', Decoder='{self.decoder_type}'")

        self.pose_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), 
            nn.Dropout2d(0.4),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), 
            nn.Dropout2d(0.4)
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 90, pose_dim)
            out = self.pose_cnn(dummy)
            self.pose_lstm_in = out.shape[1] * out.shape[3]

        self.sensor_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 1), padding=(1, 0)), 
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), 
            nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)) 
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 90, sensor_dim)
            out = self.sensor_cnn(dummy)
            self.sensor_lstm_in = out.shape[1] * out.shape[3]

        if self.rnn_type == 'gru':
            self.pose_rnn = nn.GRU(self.pose_lstm_in, hidden_size, batch_first=True, bidirectional=True)
            self.sensor_rnn = nn.GRU(self.sensor_lstm_in, hidden_size, batch_first=True, bidirectional=True)
        else:
            self.pose_rnn = nn.LSTM(self.pose_lstm_in, hidden_size, batch_first=True, bidirectional=True)
            self.sensor_rnn = nn.LSTM(self.sensor_lstm_in, hidden_size, batch_first=True, bidirectional=True)
        
        self.fusion_dim = (hidden_size * 2) + (hidden_size * 2)

        if self.fusion_type == 'attention':
            self.attention_fc = nn.Sequential(
                nn.Linear(self.fusion_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 2), 
                nn.Softmax(dim=1)
            )

        self.classifier_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        if self.decoder_type == 'deep':
            self.decoder_rnn = nn.LSTM(self.fusion_dim, 128, num_layers=2, batch_first=True, dropout=0.2)
            self.decoder_fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, pose_dim)
            )
        else:
            self.decoder_rnn = nn.LSTM(self.fusion_dim, 64, batch_first=True)
            self.decoder_fc = nn.Linear(64, pose_dim) 

        if self.uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, x_pose, x_sensor):
        p = x_pose.unsqueeze(1) 
        p = self.pose_cnn(p)
        p = p.permute(0, 2, 1, 3) 
        b, t, c, f = p.size()
        p = p.reshape(b, t, c * f)
        
        s = x_sensor.unsqueeze(1) 
        s = self.sensor_cnn(s)
        s = s.permute(0, 2, 1, 3)
        b, t, c, f = s.size()
        s = s.reshape(b, t, c * f)
        
        self.pose_rnn.flatten_parameters()
        p_out, _ = self.pose_rnn(p)
        p_feat = torch.mean(p_out, dim=1) 
        
        self.sensor_rnn.flatten_parameters()
        s_out, _ = self.sensor_rnn(s)
        s_feat = torch.mean(s_out, dim=1) 
        
        if self.fusion_type == 'attention':
            raw_concat = torch.cat([p_feat, s_feat], dim=1)
            weights = self.attention_fc(raw_concat)
            
            p_weighted = p_feat * weights[:, 0].unsqueeze(1)
            s_weighted = s_feat * weights[:, 1].unsqueeze(1)
            
            fusion_feat = torch.cat([p_weighted, s_weighted], dim=1)
        else:
            fusion_feat = torch.cat([p_feat, s_feat], dim=1)
        
        class_logits = self.classifier_head(fusion_feat)
        
        seq_len = x_pose.shape[1] 
        latent_expanded = fusion_feat.unsqueeze(1).repeat(1, seq_len, 1)
        
        dec_out, _ = self.decoder_rnn(latent_expanded) 
        recon_pose = self.decoder_fc(dec_out)          
        
        return class_logits, recon_pose