#!/bin/bash
set -e

echo "=== CompressAI Εργασία Demo ==="
mkdir -p data/img_proc
mkdir -p results
mkdir -p plots

echo
echo "Προεπεξεργασία εικόνων..."
python3 scripts/preprocess.py --input-dir data/img_raw --output-dir data/img_proc

echo
echo "Εκτέλεση CompressAI RD-curve συμπίεσης..."
python3 scripts/compress_rd.py --input-dir data/img_proc --output-csv results/results_rd_curve_kodak.csv

echo
echo "Δημιουργία γραφημάτων RD-curve..."
python3 scripts/plot_rd.py --input-csv results/results_rd_curve_kodak.csv --output-dir plots

echo
echo "Τέλος του Demo."
echo "Τα αποτελέσματα βρίσκονται στον φάκελο results και τα γραφήματα στον φάκελο plots."