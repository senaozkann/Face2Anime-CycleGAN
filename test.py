import argparse
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from gancycle import build_generator, InstanceNormalization

def load_image(image_path, img_size=(256, 256)):
    """Görseli açar, RGB'ye dönüştürür, boyutlandırır ve [-1, 1] aralığına normalize eder."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Görsel dosyası bulunamadı: {image_path}")

    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(img_size, Image.Resampling.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32)
    img_tensor = (img_array / 127.5) - 1.0
    img_tensor = np.expand_dims(img_tensor, axis=0) # (1, H, W, 3)
    return img_tensor, img_resized

def postprocess_image(tensor):
    """[-1, 1] aralığındaki tensörü [0, 255] uint8 PIL görseline dönüştürür."""
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    arr = tensor.numpy() if hasattr(tensor, 'numpy') else tensor
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def run_inference(args):
    img_size = (args.img_size, args.img_size)

    # 1. Model Mimarisi Oluşturma
    generator = build_generator(input_shape=(*img_size, 3), output_channels=3)

    # 2. Ağırlıkları Belirleme ve Yükleme
    weights_path = args.weights
    if not weights_path:
        if args.direction == "human2anime":
            weights_path = os.path.join(args.checkpoint_dir, "generator_human2anime.weights.h5")
        else:
            weights_path = os.path.join(args.checkpoint_dir, "generator_anime2human.weights.h5")

    if os.path.exists(weights_path):
        generator.load_weights(weights_path)
        print(f"[BİLGİ] Model ağırlıkları başarıyla yüklendi: {weights_path}")
    else:
        print(f"[UYARI] Belirtilen ağırlık dosyası bulunamadı: {weights_path}")
        print("[UYARI] Model eğitilmemiş rastgele ağırlıklarla çalıştırılıyor (örnek deneme).")

    # 3. Görseli İşleme ve Model Tahmini (Inference)
    print(f"[BİLGİ] '{args.input}' görseli dönüştürülüyor ({args.direction})...")
    input_tensor, input_pil = load_image(args.input, img_size)
    output_tensor = generator(input_tensor, training=False)
    output_pil = postprocess_image(output_tensor)

    # 4. Çıktıyı Kaydetme
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.side_by_side:
        # Girdi ve Çıktıyı yan yana birleştir
        combined_w = input_pil.width + output_pil.width
        combined_h = max(input_pil.height, output_pil.height)
        combined = Image.new('RGB', (combined_w, combined_h))
        combined.paste(input_pil, (0, 0))
        combined.paste(output_pil, (input_pil.width, 0))
        combined.save(args.output)
        print(f"[BAŞARILI] Yan yana karşılaştırma görseli kaydedildi: {args.output}")
    else:
        output_pil.save(args.output)
        print(f"[BAŞARILI] Dönüştürülmüş görsel kaydedildi: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="CycleGAN ile Tek Görsel Dönüştürme (Inference)")
    parser.add_argument("--input", type=str, required=True, help="Dönüştürülecek girdi görselinin dosya yolu")
    parser.add_argument("--output", type=str, default="./outputs/transformed_result.png", help="Çıktı görselinin kaydedileceği yol")
    parser.add_argument("--direction", type=str, choices=["human2anime", "anime2human"], default="human2anime", help="Dönüşüm yönü: human2anime veya anime2human")
    parser.add_argument("--weights", type=str, default=None, help="Özel model ağırlık dosyası (.weights.h5)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Ağırlıkların aranacağı varsayılan klasör")
    parser.add_argument("--img_size", type=int, default=256, help="İşlem çözünürlüğü (varsayılan: 256)")
    parser.add_argument("--side_by_side", action="store_true", default=True, help="Girdi ve çıktıyı yan yana tek görselde birleştirir")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
