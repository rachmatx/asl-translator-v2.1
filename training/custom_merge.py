import pandas as pd
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Penggunaan: python custom_merge.py <original_csv> <custom_csv>")
        sys.exit(1)

    original_csv = sys.argv[1]
    custom_csv = sys.argv[2]

    if not os.path.exists(original_csv):
        print(f"Error: {original_csv} tidak ditemukan.")
        sys.exit(1)
        
    if not os.path.exists(custom_csv):
        print(f"Error: {custom_csv} tidak ditemukan.")
        sys.exit(1)

    try:
        df_original = pd.read_csv(original_csv)
        df_custom   = pd.read_csv(custom_csv)

        # Gabungkan dan acak
        df_merged = pd.concat([df_original, df_custom], ignore_index=True)
        df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Simpan kembali ke original_csv (menimpa yang lama)
        df_merged.to_csv(original_csv, index=False)
        print(f"[OK] Berhasil menggabungkan dataset.")
        print(f"Total sampel {original_csv}: {len(df_original)}")
        print(f"Total sampel {custom_csv}: {len(df_custom)}")
        print(f"Total sampel gabungan: {len(df_merged)}")
        
    except Exception as e:
        print(f"Error saat menggabungkan data: {e}")
