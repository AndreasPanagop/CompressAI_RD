import pandas as pd
import matplotlib.pyplot as plt
import os, argparse

# Συνάρτηση για τη δημιουργία των καμπυλών Ρυθμού-Παραμόρφωσης (RD)
def generate_rd_plot(df, xcol, ycol, outpath, ylabel, title):
    plt.figure(figsize=(8, 6))
    for model in df['model'].unique():
        subset = df[df['model'] == model]
        plt.plot(subset[xcol], subset[ycol], marker='o', label=model, linewidth=2, markersize=6)
    
    # Ορίζουμε x και y labels, τίτλο και πλέγμα
    plt.xlabel(xcol.upper())
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True) # Δημιουργία φακέλου αν δεν υπάρχει
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f'Οι καμπύλες αποθηκεύτηκαν στο {outpath}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', default='results_rd_curve.csv', help='Αρχείο CSV με τα αποτελέσματα')
    parser.add_argument('--output-dir', default='plots', help='Φάκελος εξόδου για τις καμπύλες')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True) 

    # Έλεγχος αν το αρχείο CSV υπάρχει
    if not os.path.exists(args.input_csv):
        print(f"Το αρχείο {args.input_csv} δεν βρέθηκε.")
        return
    
    # Φορτώνουμε τα δεδομένα από το CSV ως DataFrame
    df = pd.read_csv(args.input_csv)
    df['psnr'] = pd.to_numeric(df['psnr'], errors='coerce')
    df['ms_ssim'] = pd.to_numeric(df['ms_ssim'], errors='coerce')
    df['bpp'] = pd.to_numeric(df['bpp'], errors='coerce')

    # Ομαδοποιούμε τα δεδομένα κατά μοντέλο και επίπεδο ποιότητας, υπολογίζοντας τον μέσο όρο
    g = df.groupby(['model', 'level']).agg({'bpp': 'mean', 'psnr': 'mean', 'ms_ssim': 'mean'}).reset_index()

    # Δημιουργούμε και αποθηκεύουμε τις καμπύλες RD για PSNR και MS-SSIM
    generate_rd_plot(g, 'bpp', 'psnr', os.path.join(args.output_dir, 'rd_curve_psnr.png'), 'PSNR (dB)', 'Καμπύλη R-D (PSNR)')
    generate_rd_plot(g, 'bpp', 'ms_ssim', os.path.join(args.output_dir, 'rd_curve_ms_ssim.png'), 'MS-SSIM', 'Καμπύλη R-D (MS-SSIM)')

    print("\nΟλοκληρώθηκε η δημιουργία των καμπυλών R-D.")
  
if __name__ == '__main__':
    main()