from glob import glob
from pathlib import Path
import os, csv, time, math, torch, argparse
import numpy as np
from PIL import Image
from pytorch_msssim import ms_ssim
from compressai.zoo import cheng2020_attn, bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean

# Η συνάρτηση φόρτωσης εικόνας και μετατροπής της σε φορμά tensor (πολυδιάστατος πίνακας pixel)
def load_image_tensor(image_path, device):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img

# Με την παρακάτω συνάρτηση υπολογίζουμε το PSNR (Peak Signal-to-Noise Ratio)
def compute_psnr(x, x_hat):
    mse = torch.mean((x - x_hat) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1 / mse)

# Εδώ υπολογίζουμε τα bits ανά pixel (bpp)
def compute_bpp(strings, H, W):
    total_bits = sum(len(s[0]) * 8 for s in strings)
    bpp = total_bits / (H * W)
    return bpp

# Και εδώ το MS-SSIM (Multi-Scale Structural Similarity)
def compute_msssim(x, x_hat):
    return ms_ssim(x, x_hat, data_range=1.0, size_average=True).item()

def main():
    # Δηλώνουμε τα ορίσματα όπως και στο preprocess.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/img_proc', help='Φάκελος με τις έτοιμες εικόνες')
    parser.add_argument('--output-csv', default='results_rd_curve.csv', help='Εξαγωγή αρχείου CSV με τα αποτελέσματα')
    parser.add_argument('--device', default='cpu', help='Συσκευή για εκτέλεση.')
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True) # Δημιουργία φακέλου αν δεν υπάρχει

    qualities = [2, 3, 4, 5, 6] # Επίπεδα ποιότητας 

    # Παρακάτω ορίζουμε τα 4 μοντέλα που θα χρησιμοποιήσουμε οπως στο παράδειγμα του PDF
    models = [
        ('cheng2020_attn', lambda q: cheng2020_attn(quality=q, pretrained=True).eval()),
        ('bmshj2018_factorized', lambda q: bmshj2018_factorized(quality=q, pretrained=True).eval()),
        ('bmshj2018_hyperprior', lambda q: bmshj2018_hyperprior(quality=q, pretrained=True).eval()),
        ('mbt2018_mean', lambda q: mbt2018_mean(quality=q, pretrained=True).eval()),
    ]

    img_paths = []
    for ext in ('png', 'jpg', 'jpeg'):
        img_paths.extend(glob(os.path.join(args.input_dir, f'*.{ext}'))) # Συλλογή όλων των πιθανών εικόνων png, jpg, jpeg
    img_paths = sorted(img_paths) # Ταξινόμηση των εικόνων
    if not img_paths:
        print(f"Δεν βρέθηκαν εικόνες στον φάκελο {args.input_dir}.")
        return
    
    header = ['img', 'model', 'level', 'bpp', 'psnr', 'ms_ssim', 'enc_time', 'dec_time', 'mem'] # Ορισμός κεφαλίδας CSV

    with open(args.output_csv, 'w', newline='') as csvfile:
        # Καλούμε τον writer για να γραφούν τα αποτελέσματα σε αρχείο .csv
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for model_name, build in models:
            for q in qualities:
                print(f"\n=== {model_name} | quality={q} ===")
                model = build(q).to(device).eval()

                for img_path in img_paths:
                    img_name = os.path.basename(img_path)
                    print(f"Επεξεργασία εικόνας: {img_name}")

                    try:
                        x = load_image_tensor(img_path, device)
                    except Exception as e:
                        print(f'Σφάλμα φόρτωσης εικόνας {img_name}: {e}')
                        continue

                    _, _, H, W = x.shape # Λήψη διαστάσεων εικόνας

                    # Compress και Decompress μετρώντας τον χρόνο
                    t0 = time.time()
                    with torch.no_grad():
                        out_enc = model.compress(x)
                    t1 = time.time()
                    enc_time = t1 - t0

                    strings = out_enc['strings']
                    bpp = compute_bpp(strings, H, W)

                    t0 = time.time()
                    with torch.no_grad():
                        out_dec = model.decompress(out_enc['strings'], out_enc['shape'])
                    t1 = time.time()
                    dec_time = t1 - t0

                    # Υπολογισμός PSNR και MS-SSIM
                    x_hat = out_dec['x_hat']
                    psnr = compute_psnr(x, x_hat)
                    msssim = compute_msssim(x, x_hat)
                    
                    writer.writerow([img_name, model_name, q, f'{bpp:.4f}', f'{psnr:.2f}', f'{msssim:.4f}', f'{enc_time:.4f}', f'{dec_time:.4f}'])
                    print(f"{img_name} | bpp: {bpp:.4f}, PSNR: {psnr:.2f}, MS-SSIM: {msssim:.4f}, Enc time: {enc_time:.4f}s, Dec time: {dec_time:.4f}s")
    print(f"\nΑποτελέσματα αποθηκεύτηκαν στο {args.output_csv}")

if __name__ == '__main__':
    main()