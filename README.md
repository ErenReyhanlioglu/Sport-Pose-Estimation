#  Sport Pose Estimation  
## Yapay Zeka Tabanlı Kişisel Antrenör

Bu proje, spor yapan bireylerin hareketlerini sadece tanımakla kalmayıp, hareketin doğru formda yapılıp yapılmadığını analiz eden Yapay Zeka Kişisel Antrenörü vizyonuyla geliştirilmiş çok görevli (Multi-Task Learning) bir derin öğrenme sistemidir.

---

##  Proje Özeti
Tek başına spor yapan bireyler genellikle profesyonel geri bildirimden yoksundur. Bu durum yanlış egzersiz formu, performans düşüşü ve sakatlanma riskini artırır.

Bu proje, klasik egzersiz sınıflandırma yaklaşımlarının ötesine geçerek hareketin ne kadar doğru yapıldığını analiz eden bir form değerlendirme sistemi sunar.

---

##  Ana Özellikler
* **Çift Görevli Mimari:**Aynı anda hem egzersiz sınıflandırması yapar hem de hareketin geometrik rekonstrüksiyonunu (yeniden oluşturma) gerçekleştirir.
* **Anomali Tespiti:** Yüksek yeniden oluşturma hatası (reconstruction error), form bozukluğunun bir göstergesi olarak kullanılır.
* **Çok Modlu Veri Füzyonu:** Poz (61 nokta) ve sensör verilerini CNN tabanlı öznitelik çıkarıcılarla işleyip LSTM/GRU ağlarında birleştirir.
* **Multi-Task Learning (MTL):** Sınıflandırma ve Autoencoder yapılarını tek bir ağda eğiterek genelleştirme yeteneğini artırır.
* **Dinamik Kayıp Dengeleme:** Belirsizlik ağırlıklandırması (Uncertainty Weighting) ile çoklu görevler arasındaki kayıp dengesini otomatik ayarlar.
* **Akıllı Geri Bildirim Sistemi:** İdeal hareket formu ile kullanıcı hareketi arasındaki sapmaları analiz ederek eklem bazlı düzeltme önerileri ("Dizi 15° kır", "Sırtı dikleştir" vb.) sunar.
* **Ablasyon Çalışması Desteği:** Konfigürasyon dosyası üzerinden farklı füzyon teknikleri (Concat/Attention) ve RNN tipleri (LSTM/GRU) kolayca test edilebilir.

---

##  Veri Seti: MM-Fit
-Projede, spor egzersizleri için özel olarak oluşturulmuş çok cihazlı ve çok modlu MM-Fit veri seti kullanılmıştır.
-İçerik: 800 dakikadan fazla sensör ve video verisi.
-Cihazlar: Akıllı saatler (bilek), akıllı telefonlar (cep) ve kulaklıklar.
-Egzersizler (10 Temel Hareket): Squats, Push-ups, Dumbbell Shoulder Presses, Lunges, Standing Dumbbell Rows, Sit-ups, Dumbbell Tricep Extensions, Bicep Curls, Sitting Dumbbell Lateral Raises ve Jumping Jacks
---

## Dosya Yapısı ve Modüller

Proje, modülerlik ve sürdürülebilirlik prensiplerine göre yapılandırılmıştır:

| Dosya | Açıklama |
| :--- | :--- |
| `main.py` | **Ana Yönetici:** Eğitim, test ve değerlendirme pipeline'ını yönetir. |
| `config.yaml` | **Konfigürasyon:** Hiperparametreler, model ayarları ve dosya yolları burada tanımlanır. |
| `model.py` | **Mimari:** `MultiModalClassifier`, `PoseAutoencoder` ve `MultiTaskNetwork` sınıflarını içerir. |
| `preprocessing.py` | **Veri İşleme:** Normalizasyon, gürültü ekleme (Noise Injection) ve Stratified Split işlemlerini yapar. |
| `trainer.py` | **Eğitim Döngüsü:** Modellerin eğitimi, validasyonu ve Erken Durdurma (Early Stopping) mantığını içerir. |
| `evaluater.py` | **Değerlendirme:** F1-Score, Accuracy, MSE ve MAE metriklerini hesaplar. |
| `inference.py` | **Analiz:** Eğitilmiş modeli kullanarak hareket düzeltme raporları üretir. |
| `plots.py` | **Görselleştirme:** Eğitim geçmişi, Confusion Matrix ve kayıp grafikleri çizer. |
| `experiment.py` | **Loglama:** Her deney için tarih damgalı klasörler oluşturur ve sonuçları saklar. |

## Kurulum

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn pyyaml tqdm

## Model Mimarisi
Encoder–Decoder tabanlı uçtan uca eğitilebilir mimari.

### Encoder
- Sensör ve Poz Dalları: 1D-CNN katmanları ile uzaysal öznitelikler çıkarılır.

-Zamansal Modelleme: RNN katmanları (GRU veya LSTM) ile hareketin zaman içindeki değişimi modellenir.

-Füzyon: İki farklı veri dalı, dikkat mekanizması (Attention) veya birleştirme (Concat) yöntemleriyle bir araya getirilir. 

### Decoder
-Classifier Head: Egzersizin sınıfını belirlemek için Fully Connected katmanlar ve Softmax kullanır.

-Reconstruction Head: Sıkıştırılmış veriden orijinal poz dizisini tekrar üreterek hareket formunu öğrenir.

---

##  Multi-Task Loss
Görevler arasındaki ağırlık dengesini otomatik olarak ayarlamak için belirsizlik ağırlıklı (uncertainty weighting) kayıp fonksiyonu kullanılmıştır:

L_total = (1 / 2σ₁²)·L_cls + (1 / 2σ₂²)·L_recon + log(σ₁σ₂)

---

## Deney Sonuçları
Proje kapsamında 6 farklı model konfigürasyonu test edilmiştir:

| Konfigürasyon | Özellik                             | Val Acc (%) | Recon MSE  |
| ------------- | ----------------------------------- | ----------- | ---------- |
| V0            | Regülarizasyonsuz                   | 90.49       | 0.2245     |
| V1            | + Uncertainty Weighting             | 90.17       | 0.2221     |
| V2            | + Regülarizasyon (Baseline)         | 88.78       | 0.2187     |
| V2.1          | Attention Fusion                    | 88.85       | 0.2185     |
| V2.2          | LSTM RNN (En iyi sınıflandırıcı)    | **91.56**   | 0.2009     |
| **V2.3**      | Deep Decoder (En iyi form analizci) | 89.04       | **0.1803** |


---

## Final Model
Form analizi açısından en iyi performansı verdiği için **V2.3 (Deep Decoder)** seçilmiştir.

---

## Proje Yapısı
```
-Proje, modüler bir Python yapısında geliştirilmiştir:

-scripts/preprocessing.py: Veri senkronizasyonu, normalizasyon ve gürültü ekleme.

-scripts/model.py: Model mimarisinin tanımlandığı ana dosya.

-scripts/trainer.py: Eğitim döngüleri ve MTL yönetimi.

-scripts/evaluater.py: Detaylı metrik raporlama ve rekonstrüksiyon analizi.

-scripts/plots.py: Eğitim geçmişi ve hata dağılım grafiklerinin oluşturulması.

scripts/
 ├─ preprocessing.py
 ├─ model.py
 ├─ trainer.py
 ├─ evaluater.py
 └─ plots.py
configs/
 └─ config.yaml
run_pipeline.py
```

---

##  Kullanım

### Sınıflandırma
```python
run_pipeline(config_path='configs/config.yaml', mode='cls')
```

### Multi-Task Eğitim
```python
run_pipeline(config_path='configs/config.yaml', mode='mtl')
```

---
