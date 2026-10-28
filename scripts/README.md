# CompressAI × uvg266 (VVC): Καμπύλες Ρυθμού-Παραμόρφωσης και “Video for Machines”

H συγκεκριμένη εργασία αξιολογεί προεκπαιδευμένα μοντέλα CompressAI παράγοντας καμπύλες ρυθμού-παραμόρφωσης (RD) για εικόνες. Η ροή των python scripts παίρνει τις εικόνες από το Kodak dataset, τις κάνει preprocess (κλιμάκωση και κοπή), τις εισάγει στα compression μοντέλα και δημιουργεί καμπύλες ρυθμού-παραμόρφωσης.

## Στήσιμο περιβάλλοντος

Σε ένα WSL terminal:
```bash
cd compressai_project/
python3 -m venv venv
source venv/bin/activate
code .
```

## Δομή Φακέλων
```
compressai_project/
   data/
      img_proc/
      img_raw/
   results/
   plots/
   scripts/
   run.sh
```

## Python Scripts

### preprocess.py
- Παίρνει ως είσοδο τις φωτογραφίες του Kodak dataset
- Εφαρμόζει resize και center crop σε αυτές
- Αποθηκεύει τις επεξεργασμένες φωτογραφίες στον φάκελο: data/img_proc

### compress_rd.py
- Φορτώνει 4 προεκπαιδευμένα μοντέλα του CompressAI
- Εκτελεί compression/decompression σε σταθερό ladder 5 επιπέδων ποιότητας (2-6)
- Υπολογίζει μετρικές: bpp, PSNR, MS-SSIM, enc_time, dec_time
- Γράφει τα αποτελέσματα σε αρχείο .csv στον φάκελο results

### plot_rd.py
- Διαβάζει το αρχείο .csv 
- Παράγει 2 καμπύλες με την βοήθεια της pyplot (rd_curve_ms_ssim.png, rd_curve_psnr.png)
- Αποθηκεύει τις καμπύλες στον φάκελο plots

## Εκτέλεση της ροής με μία εντολή (run.sh)
```bash
chmod +x run.sh
./run.sh
```
O bash κώδικας εκτελεί τα 3 scripts με την σειρά που αναγράφονται παραπάνω.

## BONUS 2 Scripts

### bonus2.py 
- Μετράει χρόνους κωδικοποίησης/αποκωδικοποίησης και μνήμης για 2–3 μοντέλα/επίπεδα (μόνο για τις εικόνες Kodak)

### bonus2_plot.py 
- Με είσοδο το αποτέλεσμα του προηγούμενου script (bonus2.csv) παράγει διάγραμμα με καμπύλες.