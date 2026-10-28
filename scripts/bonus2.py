from pathlib import Path
from PIL import Image
from compressai.zoo import cheng2020_attn, bmshj2018_factorized
import os, csv, time, torch, argparse, psutil, subprocess
import numpy as np

def load_image_tensor(image_path, device):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img

# Μετράμε μόνο την χρήση μνήμης του επεξεργαστή καθώς δεν χρησιμοποιούμε GPU
def measure_ram():
    return psutil.Process().memory_info().rss / (1024 ** 2)

# Δοκιμάζουμε την κωδικοποίηση και αποκωδικοποίηση με το δοσμένο μοντέλο
def test_compressai(model, x):
    cpu0 = measure_ram()

    t0 = time.time()
    out = model.compress(x)
    t_enc = time.time() - t0

    cpu1 = measure_ram()

    t0 = time.time()
    _ = model.decompress(out['strings'], out['shape'])
    t_dec = time.time() - t0

    cpu2 = measure_ram()
    cpu_mb = max(cpu0, cpu1, cpu2)

    return t_enc, t_dec, cpu_mb

def compute_bpp(strings, H, W):
    total_bits = sum(len(s[0]) * 8 for s in strings)
    bpp = total_bits / (H * W)
    return bpp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', default='data/img_proc')
    parser.add_argument('--output-csv', default='results/bonus2.csv')
    args = parser.parse_args()

    images = sorted(Path(args.images_dir).glob('*.png'))
    if not images:
        print(f"Δεν βρέθηκαν εικόνες στον φάκελο {args.images_dir}.")
        return
    
    x = load_image_tensor(images[0], 'cpu')
    _, H, W = x.shape[1:]

    models = [
        ('cheng2020_attn', lambda q: cheng2020_attn(quality=q, pretrained=True).eval()),
        ('bmshj2018_factorized', lambda q: bmshj2018_factorized(quality=q, pretrained=True).eval()),
    ]

    qualities = [2, 4, 6]

    header = ['method', 'model', 'quality', 'bitrate', 'enc_time', 'dec_time', 'mem_cpu_MB']

    with open(args.output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for name, build in models:
            for q in qualities:
                model = build(q)
                model = model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
                t_enc, t_dec, cpu_mb = test_compressai(model, x)
                out = model.compress(x)
                bpp = compute_bpp(out['strings'], H, W)
                writer.writerow(['compressai', name, q, f"{bpp:.4f}", f"{t_enc:.2f}", f"{t_dec:.2f}", f"{cpu_mb:.2f}"])
                print(f"{name} | quality={q} | bpp={bpp:.4f} | enc_time={t_enc:.2f} s | dec_time={t_dec:.2f} s")

       
if __name__ == '__main__':
    main()

           