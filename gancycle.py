import argparse
import glob
import os
import random
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model

# -----------------------------------------------------------------------------
# 1. Donanım (GPU/CPU) Yapılandırması
# -----------------------------------------------------------------------------
def setup_device():
    """Mevcut GPU'ları tespit eder ve bellek büyümesini dinamik olarak ayarlar."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] {len(gpus)} GPU tespit edildi. Dinamik bellek büyümesi aktifleştirildi.")
        except RuntimeError as e:
            print(f"[UYARI] GPU bellek yapılandırması hatası: {e}")
    else:
        print("[UYARI] GPU bulunamadı! Kod CPU üzerinde çalışacak (CPU eğitimi yavaş olabilir).")

# -----------------------------------------------------------------------------
# 2. Özel Katmanlar: Instance Normalization & Residual Block
# -----------------------------------------------------------------------------
class InstanceNormalization(layers.Layer):
    """
    Instance Normalization Katmanı (Ulyanov et al., 2016).
    CycleGAN'da Batch Normalization yerine kullanılır.
    Harici bağımlılık gerektirmeden tüm modern TensorFlow sürümleriyle uyumludur.
    """
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        depth = input_shape[-1]
        self.scale = self.add_weight(
            name='scale',
            shape=(depth,),
            initializer=tf.random_normal_initializer(1.0, 0.02),
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=(depth,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv
        return self.scale * normalized + self.offset


class ResidualBlock(layers.Layer):
    """
    CycleGAN ResNet Bloğu:
    Conv2D -> InstanceNorm -> ReLU -> Conv2D -> InstanceNorm + Skip Connection
    """
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)
        self.in1 = InstanceNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)
        self.in2 = InstanceNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.in1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.in2(x)
        return x + inputs

# -----------------------------------------------------------------------------
# 3. Model Mimarileri: Generator (ResNet-9) & Discriminator (PatchGAN 70x70)
# -----------------------------------------------------------------------------
def build_generator(input_shape=(256, 256, 3), output_channels=3, n_residual_blocks=9):
    """
    CycleGAN Generator Mimarisi (ResNet Tabanlı):
    c7s1-64 -> d128 -> d256 -> 9 x R256 -> u128 -> u64 -> c7s1-3 (tanh)
    """
    inputs = layers.Input(shape=input_shape)

    # Başlangıç Evrişimi: 7x7 conv, 64 filtre
    x = layers.Conv2D(64, (7, 7), padding='same', use_bias=False)(inputs)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Downsampling (Örnekleme Küçültme)
    # 64 -> 128 (Boyut: 128x128)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # 128 -> 256 (Boyut: 64x64)
    x = layers.Conv2D(256, (3, 3), strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Residual Bloklar (256 filtre giriş, 256 filtre çıkış)
    for _ in range(n_residual_blocks):
        x = ResidualBlock(256)(x)

    # Upsampling (Örnekleme Büyütme)
    # 256 -> 128 (Boyut: 128x128)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # 128 -> 64 (Boyut: 256x256)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Çıkış Katmanı: 7x7 conv, 3 kanal, tanh aktivasyonu [-1, 1]
    outputs = layers.Conv2D(output_channels, (7, 7), padding='same', activation='tanh')(x)

    return Model(inputs, outputs, name="Generator")


def build_discriminator(input_shape=(256, 256, 3)):
    """
    PatchGAN 70x70 Discriminator Mimarisi:
    C64 (strides=2) -> C128 (strides=2) -> C256 (strides=2) -> C512 (strides=1) -> C1 (strides=1)
    """
    inputs = layers.Input(shape=input_shape)

    # Katman 1: Norm uygulanmaz
    x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)

    # Katman 2: 128 filtre
    x = layers.Conv2D(128, (4, 4), strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Katman 3: 256 filtre
    x = layers.Conv2D(256, (4, 4), strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Katman 4: 512 filtre, stride=1 (PatchGAN detayı)
    x = layers.Conv2D(512, (4, 4), strides=1, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Çıkış: 1 kanal, lineer aktivasyon (LSGAN için doğrudan logitler)
    outputs = layers.Conv2D(1, (4, 4), strides=1, padding='same')(x)

    return Model(inputs, outputs, name="Discriminator")

# -----------------------------------------------------------------------------
# 4. Bellek Dostu Veri Yükleyici (tf.data Pipeline)
# -----------------------------------------------------------------------------
def load_and_preprocess_image(file_path, img_size=(256, 256), is_training=True):
    """Diskten görseli lazy okur, RGB'ye çevirir, boyutlandırır ve [-1, 1] aralığına normalize eder."""
    img_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0

    if is_training:
        img = tf.image.random_flip_left_right(img)

    return img


def create_dataset(image_dir, img_size=(256, 256), batch_size=1, max_samples=None, is_training=True):
    """
    Belirtilen dizindeki tüm görselleri tarar ve RAM'i tüketmeyen lazy tf.data.Dataset döndürür.
    """
    valid_exts = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG')
    file_paths = []
    for ext in valid_exts:
        file_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if not file_paths:
        raise FileNotFoundError(f"Dizinde desteklenen görsel bulunamadı: {image_dir}")

    if max_samples and max_samples < len(file_paths):
        random.shuffle(file_paths)
        file_paths = file_paths[:max_samples]

    print(f"[DATA] '{image_dir}' dizininden {len(file_paths)} görsel yüklendi.")

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), 1000))

    dataset = dataset.map(
        lambda path: load_and_preprocess_image(path, img_size, is_training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, len(file_paths)

# -----------------------------------------------------------------------------
# 5. Görsel Kaydetme Yardımcısı
# -----------------------------------------------------------------------------
def save_comparison_grid(real_a, fake_b, real_b, fake_a, filepath):
    """
    [-1, 1] aralığındaki tensörleri [0, 255] uint8'e dönüştürür ve
    karşılaştırma grid'i (2x2) olarak kaydeder:
    Üst Satır: Gerçek İnsan -> Üretilen Anime
    Alt Satır: Gerçek Anime -> Üretilen İnsan
    """
    def denorm(t):
        if len(t.shape) == 4:
            t = t[0]
        arr = t.numpy() if hasattr(t, 'numpy') else t
        arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return arr

    img_ra = denorm(real_a)
    img_fb = denorm(fake_b)
    img_rb = denorm(real_b)
    img_fa = denorm(fake_a)

    top_row = np.concatenate([img_ra, img_fb], axis=1)
    bottom_row = np.concatenate([img_rb, img_fa], axis=1)
    grid = np.concatenate([top_row, bottom_row], axis=0)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(grid).save(filepath)

# -----------------------------------------------------------------------------
# 6. CycleGAN Eğitim Döngüsü (Training Loop)
# -----------------------------------------------------------------------------
def train_cyclegan(args):
    setup_device()

    img_size = (args.img_size, args.img_size)
    output_dir = args.output_dir
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Veri Yükleyicileri
    human_ds, n_human = create_dataset(args.human_dir, img_size, args.batch_size, args.max_samples, is_training=True)
    anime_ds, n_anime = create_dataset(args.anime_dir, img_size, args.batch_size, args.max_samples, is_training=True)
    steps_per_epoch = min(n_human, n_anime) // args.batch_size

    # Sabit test örnekleri (her epoch aynı yüzün gelişimini izlemek için)
    sample_human = next(iter(human_ds))
    sample_anime = next(iter(anime_ds))

    # 2. Modeller
    G_A2B = build_generator(input_shape=(*img_size, 3), output_channels=3) # İnsan -> Anime
    G_B2A = build_generator(input_shape=(*img_size, 3), output_channels=3) # Anime -> İnsan
    D_A = build_discriminator(input_shape=(*img_size, 3))                  # İnsan Ayırt Edici
    D_B = build_discriminator(input_shape=(*img_size, 3))                  # Anime Ayırt Edici

    # 3. Optimizatörler ve Kayıp Fonksiyonları
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.5)
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.5)
    mse_loss = tf.keras.losses.MeanSquaredError()
    mae_loss = tf.keras.losses.MeanAbsoluteError()

    # 4. Checkpoint Yöneticisi
    checkpoint = tf.train.Checkpoint(
        G_A2B=G_A2B, G_B2A=G_B2A,
        D_A=D_A, D_B=D_B,
        optimizer_G=optimizer_G, optimizer_D=optimizer_D
    )
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(f"[CHECKPOINT] Son kayıt yüklendi: {checkpoint_manager.latest_checkpoint}")

    # 5. Tek Adım Eğitim Fonksiyonu (@tf.function)
    lambda_cycle = args.lambda_cycle
    lambda_id = args.lambda_identity

    @tf.function
    def train_step(real_A, real_B):
        with tf.GradientTape(persistent=True) as tape:
            # İleri Besleme (Forward Passes)
            fake_B = G_A2B(real_A, training=True)
            cycled_A = G_B2A(fake_B, training=True)

            fake_A = G_B2A(real_B, training=True)
            cycled_B = G_A2B(fake_A, training=True)

            # Identity Dönüşümleri (Renk ve Yapı Koruma)
            same_B = G_A2B(real_B, training=True)
            same_A = G_B2A(real_A, training=True)

            # Discriminator Değerlendirmeleri
            disc_real_A = D_A(real_A, training=True)
            disc_fake_A = D_A(fake_A, training=True)
            disc_real_B = D_B(real_B, training=True)
            disc_fake_B = D_B(fake_B, training=True)

            # --- Generator Kayıpları ---
            # 1. Adversarial Loss (LSGAN)
            loss_g_a2b = mse_loss(tf.ones_like(disc_fake_B), disc_fake_B)
            loss_g_b2a = mse_loss(tf.ones_like(disc_fake_A), disc_fake_A)

            # 2. Cycle Consistency Loss
            loss_cycle = mae_loss(real_A, cycled_A) + mae_loss(real_B, cycled_B)

            # 3. Identity Loss
            loss_identity = mae_loss(real_A, same_A) + mae_loss(real_B, same_B)

            # Toplam Generator Kaybı
            total_loss_g = loss_g_a2b + loss_g_b2a + (lambda_cycle * loss_cycle) + (lambda_id * loss_identity)

            # --- Discriminator Kayıpları ---
            loss_d_a_real = mse_loss(tf.ones_like(disc_real_A), disc_real_A)
            loss_d_a_fake = mse_loss(tf.zeros_like(disc_fake_A), disc_fake_A)
            loss_d_a = (loss_d_a_real + loss_d_a_fake) * 0.5

            loss_d_b_real = mse_loss(tf.ones_like(disc_real_B), disc_real_B)
            loss_d_b_fake = mse_loss(tf.zeros_like(disc_fake_B), disc_fake_B)
            loss_d_b = (loss_d_b_real + loss_d_b_fake) * 0.5

        # Gradyanlar ve Ağırlık Güncellemesi
        grads_g = tape.gradient(total_loss_g, G_A2B.trainable_variables + G_B2A.trainable_variables)
        grads_d_a = tape.gradient(loss_d_a, D_A.trainable_variables)
        grads_d_b = tape.gradient(loss_d_b, D_B.trainable_variables)

        optimizer_G.apply_gradients(zip(grads_g, G_A2B.trainable_variables + G_B2A.trainable_variables))
        optimizer_D.apply_gradients(zip(grads_d_a, D_A.trainable_variables))
        optimizer_D.apply_gradients(zip(grads_d_b, D_B.trainable_variables))

        del tape
        return total_loss_g, loss_d_a, loss_d_b

    # 6. Eğitim Döngüsü
    print(f"\n[EĞİTİM BAŞLIYOR] Toplam Epoch: {args.epochs} | Adım/Epoch: {steps_per_epoch}\n")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        step = 0
        total_g_loss = 0.0
        total_da_loss = 0.0
        total_db_loss = 0.0

        for real_A, real_B in tf.data.Dataset.zip((human_ds, anime_ds)):
            loss_g, loss_da, loss_db = train_step(real_A, real_B)
            total_g_loss += loss_g.numpy()
            total_da_loss += loss_da.numpy()
            total_db_loss += loss_db.numpy()
            step += 1

            if step % 20 == 0 or step == steps_per_epoch:
                print(f"\rEpoch [{epoch}/{args.epochs}] Adım [{step}/{steps_per_epoch}] "
                      f"| G Loss: {loss_g:.4f} | D_A Loss: {loss_da:.4f} | D_B Loss: {loss_db:.4f}", end="")

        elapsed = time.time() - start_time
        avg_g = total_g_loss / max(step, 1)
        avg_da = total_da_loss / max(step, 1)
        avg_db = total_db_loss / max(step, 1)
        print(f"\n[Epoch {epoch} Tamamlandı - {elapsed:.1f} sn] Ortalama G: {avg_g:.4f} | D_A: {avg_da:.4f} | D_B: {avg_db:.4f}")

        # Karşılaştırma görseli üret ve kaydet
        if epoch % args.save_freq == 0:
            test_fake_b = G_A2B(sample_human, training=False)
            test_fake_a = G_B2A(sample_anime, training=False)
            output_path = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
            save_comparison_grid(sample_human, test_fake_b, sample_anime, test_fake_a, output_path)
            print(f"[KAYDEDİLDİ] Örnek görsel kaydedildi: {output_path}")

        # Checkpoint kaydet
        if epoch % args.checkpoint_freq == 0:
            saved_path = checkpoint_manager.save()
            print(f"[CHECKPOINT] Model ağırlıkları kaydedildi: {saved_path}")

    # Eğitim sonu nihai model kaydı
    final_a2b_path = os.path.join(checkpoint_dir, "generator_human2anime.weights.h5")
    final_b2a_path = os.path.join(checkpoint_dir, "generator_anime2human.weights.h5")
    G_A2B.save_weights(final_a2b_path)
    G_B2A.save_weights(final_b2a_path)
    print(f"\n[BAŞARILI] Eğitim tamamlandı. Nihai ağırlıklar kaydedildi:\n- {final_a2b_path}\n- {final_b2a_path}")

# -----------------------------------------------------------------------------
# 7. Komut Satırı Argümanları & Giriş Noktası
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CycleGAN ile İnsan <-> Anime Dönüşümü")
    parser.add_argument("--human_dir", type=str, default="./insan", help="İnsan yüzleri veri klasörü")
    parser.add_argument("--anime_dir", type=str, default="./anime", help="Anime yüzleri veri klasörü")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Örnek görsel çıktılarının kaydedileceği klasör")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Model ağırlıklarının kaydedileceği klasör")
    parser.add_argument("--epochs", type=int, default=50, help="Toplam eğitim epoch sayısı (varsayılan: 50)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch boyutu (CycleGAN için standart: 1)")
    parser.add_argument("--img_size", type=int, default=256, help="Görsel çözünürlüğü (varsayılan: 256)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Öğrenme oranı (Learning rate, varsayılan: 0.0002)")
    parser.add_argument("--lambda_cycle", type=float, default=10.0, help="Cycle consistency loss ağırlığı (varsayılan: 10.0)")
    parser.add_argument("--lambda_identity", type=float, default=5.0, help="Identity loss ağırlığı (varsayılan: 5.0)")
    parser.add_argument("--max_samples", type=int, default=None, help="Eğitimde kullanılacak maksimum görsel sayısı (örn: hızlı deneme için 500)")
    parser.add_argument("--save_freq", type=int, default=1, help="Görsel kaydetme periyodu (epoch cinsinden)")
    parser.add_argument("--checkpoint_freq", type=int, default=5, help="Model ağırlığı kaydetme periyodu (epoch cinsinden)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_cyclegan(args)
