from PIL import Image
from pathlib import Path
import torchvision.transforms as T
import argparse

long_side = 768
crop_size = 512

# Εδώ παραθέτουμε την συνάρτηση κοπής όπως ήταν στο Project briefing PDF για το center crop.
def resize_long_then_center_crop(img_pil):
    w, h = img_pil.size
    if w >= h:
        new_w = long_side
        new_h = int(h * (long_side / w))
    else:
        new_h = long_side
        new_w = int(w * (long_side / h))
    tfm = T.Compose([
    T.Resize((new_h, new_w), interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop((crop_size, crop_size)),
    ])

    return tfm(img_pil)

def main():
    # Όπως θα δούμε και στα επόμενα scripts, χρησιμοποιούμε το argparse για να έχουμε μεγαλύτερη άνεση 
    # στα ορίσματα όταν εκτελούμε το script από το WSL terminal.
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True, help="Ο φάκελος με τα raw frames.")
    parser.add_argument('--output-dir', required=True, help="Ο φάκελος που θα αποθηκευτούν τα επεξεργασμένα frames.")
    args = parser.parse_args()

    # Μετατρέπουμε τα ορίσματα εισόδου και εξόδου σε αντικείμενα Path.
    i = Path(args.input_dir)
    o = Path(args.output_dir)
    o.mkdir(parents=True, exist_ok=True)

    # Με το glob(), παίρνουμε όλες τις png εικόνες από τον φάκελο εισόδου και στη συνέχεια τις ταξινομούμε με το sorted().
    frames = sorted(i.glob('*.png'))
    if not frames:
        print("Δεν βρέθηκαν αρχεία PNG στον φάκελο εισόδου.")
        return
    
    # Κάθε εικόνα μετατρέπεται σε RGB (3 κανάλια χρώματος), και στη συνέχεια γίνεται η κοπή και αποθήκευση.
    for f in frames:
        img = Image.open(f).convert('RGB')
        proc_img = resize_long_then_center_crop(img)
        proc_img.save(o / f.name, format='PNG')
    
    print("Τέλος προεπεξεργασίας.")

if __name__ == "__main__":
    main()

