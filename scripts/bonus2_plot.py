import matplotlib.pyplot as plt
import pandas as pd

# Φορτώνουμε τα δεδομένα από το αρχείο CSV
df = pd.read_csv('results/bonus2.csv')

plt.figure(figsize=(8, 6))

models = df['model'].unique()

# Σχεδιάζουμε τις καμπύλες για κάθε μοντέλο
for model in models:
    subset = df[df['model']==model]
    plt.plot(subset['quality'], subset['enc_time'], marker='o', label=f'{model}  Χρόνος Κωδικοποίησης')
    plt.plot(subset['quality'], subset['dec_time'], marker='s', label=f'{model} Χρόνος Αποκωδικοποίησης')

# Προσθέτουμε ετικέτες και τίτλο, καθώς και το γενικό layout του γραφήματος
plt.xlabel('Επίπεδο Ποιότητας')
plt.ylabel('Χρόνος (s)')
plt.title('BONUS 2: Κωδικοποίηση και Αποκωδικοποίηση Χρόνου ανά Επίπεδο Ποιότητας')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('plots/bonus2.png', dpi=200)
print(f"Το γράφημα αποθηκεύτηκε στο plots/bonus2.png")

