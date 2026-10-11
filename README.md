<div align="center">

# 🎨 CycleGAN: Human Face <-> Anime Face Translation

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Eşleştirilmemiş (Unpaired) İnsan ve Anime Yüzleri Arasında Karşılıklı Stil Transferi Yapan Derin Öğrenme Modeli**

<br/>

<img src="assets/sample_showcase.png" alt="CycleGAN Human to Anime Showcase" width="750" style="border-radius: 8px;"/>

*Gerçek İnsan Yüzü $\leftrightarrow$ Anime Portresi Dönüşümü (Side-by-Side Comparison)*

</div>

---

## 🌟 Proje Özeti

Bu proje, Jun-Yan Zhu ve ekibinin öncülüğünü yaptığı **Cycle-Consistent Generative Adversarial Networks (CycleGAN)** mimarisine dayanmaktadır. Geleneksel modellerin aksine, aynı kişinin hem gerçek hem de anime çiziminin bulunmasına gerek kalmadan (**unpaired dataset**) iki farklı görsel alanı (domain) arasında yüksek kaliteli stil transferi gerçekleştirir:

- **$G_{A \to B}$ (Human $\to$ Anime):** Gerçek insan fotoğrafını Japon anime estetiğinde çizime dönüştürür.
- **$G_{B \to A}$ (Anime $\to$ Human):** Anime karakter çizimini gerçekçi insan yüzü özelliklerine uyarlar.

---

## 🖼️ Model Çıktıları ve Karşılaştırma

Aşağıdaki tabloda modelin dönüştürme sonuçlarının şematik yapısı gösterilmektedir:

| Girdi (Gerçek İnsan) | Çıktı ($G_{A \to B}$ Anime) | Girdi (Anime Portresi) | Çıktı ($G_{B \to A}$ İnsan) |
| :---: | :---: | :---: | :---: |
| *(Orijinal Fotoğraf)* | *(Üretilen Çizim)* | *(Orijinal Anime)* | *(Gerçekçi Portre)* |

> Eğitim sırasında her epoch sonunda üretilen 2x2 karşılaştırma görselleri otomatik olarak `outputs/epoch_XXX.png` olarak kaydedilir.

---

## 🚀 Öne Çıkan Teknik Özellikler

- **ResNet-9 Generator**: Derin katmanlarda bilgi kaybını önleyen 9 adet Residual Block içeren üreteç mimarisi.
- **70x70 PatchGAN Discriminator**: Piksel piksel değil, yerel doku ve desen parçalarını analiz ederek yüksek keskinlik sağlayan ayırt edici.
- **Identity Loss ($\lambda_{id} = 5.0$)**: Cilt rengi, saç tonu ve arka plan kompozisyonunun bozulmasını önleyen kimlik koruma kaybı.
- **Cycle Consistency Loss ($\lambda_{cyc} = 10.0$)**: İki yönlü çevrim tutarlılığı ($A \to B \to A \approx A$).
- **Bellek Dostu (Lazy Loading) `tf.data` Pipeline**: Binlerce görseli RAM'e yığmadan, GPU eğitildikçe diskten çeken akıllı veri hattı.
- **Dahili Instance Normalization**: Harici kütüphanelere bağımlı olmayan, modern TensorFlow ile tam uyumlu özel katman.

---

## 📁 Dizin Yapısı

```text
├── assets/                  # README ve tanıtım görselleri
│   └── sample_showcase.png  # Öncesi / sonrası vitrin görseli
├── anime/                   # Anime yüzleri veri seti (.jpg, .png)
├── insan/                   # Gerçek insan yüzleri veri seti (.jpg, .png)
├── outputs/                 # Eğitim ve test çıktıları
├── checkpoints/             # Model ağırlıkları (.weights.h5)
├── gancycle.py              # Eğitim betiği ve model mimarisi
├── test.py                  # Tek görsel dönüştürme (Inference) betiği
├── requirements.txt         # Gerekli Python bağımlılıkları
├── .gitignore               # Büyük veri setlerini yok sayma kuralları
└── README.md                # Proje dokümantasyonu
```

---

## 🛠️ Kurulum

1. Repoyu klonlayın:
```bash
git clone https://github.com/KULLANICI_ADINIZ/animeokey.git
cd animeokey
```

2. Sanal ortam oluşturup etkinleştirin (Önerilir):
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate
```

3. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

---

## 🧪 Kendi Fotoğrafınızı Dönüştürme (Inference)

Eğitilmiş model ağırlıklarını kullanarak kendi fotoğrafınızı tek bir komutla animeye çevirebilirsiniz:

```bash
# İnsan fotoğrafını animeye çevirme (Yan yana karşılaştırma çıktısı verir):
python test.py --input benim_fotografim.jpg --direction human2anime

# Anime portresini insana çevirme:
python test.py --input anime_cizim.jpg --direction anime2human
```

#### Test Parametreleri:
- `--input`: Dönüştürülecek görselin dosya yolu (Zorunlu).
- `--output`: Çıktının kaydedileceği yer (Varsayılan: `outputs/transformed_result.png`).
- `--direction`: `human2anime` veya `anime2human`.
- `--weights`: Özel ağırlık dosyası yolu (Varsayılan: `checkpoints/` altındaki ağırlıklar).
- `--side_by_side`: Girdi ve çıktıyı yan yana birleştirir (Varsayılan: Açık).

---

## 🏋️ Modeli Eğitme (Training)

### Varsayılan Ayarlarla Başlatma:
```bash
python gancycle.py
```

### Özelleştirilmiş Parametrelerle Başlatma:
```bash
python gancycle.py --epochs 100 --batch_size 1 --img_size 256 --lr 0.0002 --human_dir ./insan --anime_dir ./anime
```

### Eğitim Argümanları:

| Argüman | Varsayılan | Açıklama |
|---|---|---|
| `--epochs` | `50` | Toplam eğitim turu sayısı |
| `--batch_size` | `1` | Batch boyutu (CycleGAN için standart: 1) |
| `--img_size` | `256` | Çözünürlük (256x256) |
| `--lr` | `0.0002` | Adam optimizatörü öğrenme oranı |
| `--lambda_cycle` | `10.0` | Cycle consistency loss çarpanı |
| `--lambda_identity` | `5.0` | Identity loss çarpanı |
| `--max_samples` | `None` | Hızlı deneme için maksimum görsel sayısı (örn. `500`) |
| `--save_freq` | `1` | Karşılaştırma görseli kaydetme sıklığı (epoch) |
| `--checkpoint_freq` | `5` | Model ağırlıklarını kaydetme sıklığı (epoch) |

---

## 📜 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.
