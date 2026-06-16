import pandas as pd
import numpy as np

# ==============================================================================
# TUGAS BESAR KECERDASAN BUATAN
# Sistem Klasifikasi dan Rekomendasi Kelayakan Air Minum Menggunakan KNN
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. FUNGSI KNN
# ------------------------------------------------------------------------------
def euclidean_distance(X_train, test_row):
    """
    Menghitung jarak Euclidean antara 1 data uji (test_row) dengan seluruh data latih (X_train).
    Menggunakan fitur broadcasting numpy agar komputasi jauh lebih cepat.
    """
    distances = np.sqrt(np.sum((X_train - test_row) ** 2, axis=1))
    return distances

def get_neighbors(distances, y_train, k):
    """Mencari K tetangga terdekat berdasarkan jarak terkecil"""
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]
    return nearest_labels, nearest_indices

def majority_voting(nearest_labels):
    """Melakukan voting mayoritas dari label tetangga terdekat"""
    labels_list = nearest_labels.tolist()
    prediction = max(set(labels_list), key=labels_list.count)
    return prediction

def predict_knn(X_train, y_train, X_test, k):
    """Fungsi prediksi untuk seluruh data uji"""
    predictions = []
    for test_row in X_test:
        distances = euclidean_distance(X_train, test_row)
        nearest_labels, _ = get_neighbors(distances, y_train, k)
        pred = majority_voting(nearest_labels)
        predictions.append(pred)
    return np.array(predictions)

# ------------------------------------------------------------------------------
# 2. FUNGSI EVALUASI 
# ------------------------------------------------------------------------------
def evaluate_model(y_true, y_pred):
    """Menghitung Accuracy dan Confusion Matrix (TP, TN, FP, FN)"""
    TP, TN, FP, FN = 0, 0, 0, 0
    correct = 0
    
    for actual, pred in zip(y_true, y_pred):
        if actual == pred:
            correct += 1
            
        if actual == 1 and pred == 1:
            TP += 1
        elif actual == 0 and pred == 0:
            TN += 1
        elif actual == 0 and pred == 1:
            FP += 1
        elif actual == 1 and pred == 0:
            FN += 1
            
    accuracy = correct / len(y_true)
    return accuracy, TP, TN, FP, FN

def main():
    print("="*75)
    print("SISTEM KLASIFIKASI DAN REKOMENDASI KELAYAKAN AIR MINUM")
    print("MENGGUNAKAN ALGORITMA K-NEAREST NEIGHBOR (KNN)")
    print("="*75)

    # --------------------------------------------------------------------------
    # TAHAP 1: PREPROCESSING DATA
    # --------------------------------------------------------------------------
    print("\n[1] Membaca Dataset dan Preprocessing...")
    try:
        # Membaca dataset
        df = pd.read_csv('water_potability.csv')
    except FileNotFoundError:
        print("ERROR: File 'water_potability.csv' tidak ditemukan!")
        return

    # Mengecek missing value sebelum diisi
    missing_before = df.isnull().sum().sum()
    print(f"    -> Ditemukan total {missing_before} missing value.")
    
    # Mengisi missing value dengan rata-rata (Mean Imputation)
    df.fillna(df.mean(), inplace=True)
    
    # Mengecek dan menghapus data duplikat
    duplikat = df.duplicated().sum()
    print(f"    -> Ditemukan {duplikat} data duplikat.")
    if duplikat > 0:
        df.drop_duplicates(inplace=True)

    # Memisahkan atribut fitur (X) dan target label (y)
    X = df.drop('Potability', axis=1).values
    y = df['Potability'].values
    fitur_names = df.drop('Potability', axis=1).columns.tolist()

    # --- ATURAN KAMUS SKALA RESMI (BERDASARKAN DATASET & ATURAN FISIS) ---
    skala_rules = {
        'ph': (0.00, 14.00, "Skala Formal 0-14"),
        'hardness': (47.43, 323.12, "mg/L"),
        'solids': (320.94, 61227.20, "ppm"),
        'chloramines': (0.35, 13.13, "ppm"),
        'sulfate': (129.00, 481.03, "mg/L"),
        'conductivity': (181.48, 753.34, "µS/cm"),
        'organic_carbon': (2.20, 28.30, "ppm"),
        'trihalomethanes': (0.74, 124.00, "µg/L"),
        'turbidity': (1.45, 6.74, "NTU")
    }

    # Membagi data menjadi training (80%) dan testing (20%) SEBELUM normalisasi
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.8)
    
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    X_train_raw, X_test_raw = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Normalisasi Min-Max pada data latih
    X_min = X_train_raw.min(axis=0)
    X_max = X_train_raw.max(axis=0)
    
    # Mencegah Division by Zero jika ada fitur dengan max == min
    range_values = X_max - X_min
    range_values[range_values == 0] = 1
    
    # Terapkan normalisasi ke Data Latih dan Data Uji menggunakan nilai dari Data Latih
    X_train = (X_train_raw - X_min) / range_values
    X_test = (X_test_raw - X_min) / range_values

    print(f"    -> Normalisasi selesai. Data Latih: {len(X_train)} baris, Data Uji: {len(X_test)} baris.")

    # --------------------------------------------------------------------------
    # TAHAP 2: PENGUJIAN K-NEAREST NEIGHBOR
    # --------------------------------------------------------------------------
    print("\n[2] Mengevaluasi Hyperparameter K...")
    k_values = [1, 3, 5, 7, 9]
    best_k = -1
    best_acc = 0.0
    best_cm = None

    for k in k_values:
        y_pred = predict_knn(X_train, y_train, X_test, k)
        acc, TP, TN, FP, FN = evaluate_model(y_test, y_pred)
        
        print(f"    -> Untuk K={k} | Akurasi: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_cm = (TP, TN, FP, FN)

    TP, TN, FP, FN = best_cm
    print(f"\n[HASIL TERBAIK] Nilai K terbaik adalah K={best_k} dengan Akurasi {best_acc*100:.2f}%\n")
    print("Confusion Matrix:")
    print(f"  [TN: {TN:<4} | FP: {FP:<4}]")
    print(f"  [FN: {FN:<4} | TP: {TP:<4}]")

    # --------------------------------------------------------------------------
    # TAHAP 3: REKOMENDASI VIA TERMINAL INTERAKTIF
    # --------------------------------------------------------------------------
    print("\n" + "="*75)
    print("INPUT PENGGUNA UNTUK PREDIKSI AIR BARU")
    print("="*75)

    while True:
        user_data = []
        batal = False # Bendera (flag) untuk mengecek apakah user ingin keluar
        
        print("\nSilakan masukkan nilai parameter kualitas air (Ketik 'q' untuk keluar):")
        
        for fitur in fitur_names:
            # Normalisasi nama variabel kunci untuk pencarian di kamus aturan skala
            key_lookup = fitur.lower().replace(" ", "_")
            
            # Ambil batas minimum dan maksimum berdasarkan data yang diberikan
            if key_lookup in skala_rules:
                min_skala, max_skala, unit = skala_rules[key_lookup]
            else:
                # Fallback cadangan dinamis jika ada ketidakcocokan nama kolom asli
                min_skala = df[fitur].min()
                max_skala = df[fitur].max()
                unit = ""
            
            while True:
                val = input(f"Masukkan nilai {fitur} (Valid: {min_skala:.2f} s/d {max_skala:.2f} {unit}): ")
                
                # Jika user mengetik 'q' atau 'Q', aktifkan bendera batal keluar dari loop input
                if val.lower() == 'q':
                    batal = True
                    break
                
                try:
                    val_float = float(val)
                    
                    # --- PROSES VALIDASI BATASAN SKALA YANG KETAT ---
                    if not (min_skala <= val_float <= max_skala):
                        print(f"    [MELEWATI BATAS] Nilai yang dimasukkan kelewatan! {fitur} harus berada di rentang {min_skala:.2f} s/d {max_skala:.2f} {unit}.")
                        continue
                    
                    # Jika lolos pengecekan skala, simpan nilai parameter
                    user_data.append(val_float)
                    break
                    
                except ValueError:
                    print("    [ERROR] Masukan tidak valid! Harus berupa angka (gunakan titik '.' untuk desimal).\n")
            
            if batal:
                break
        
        # Jika proses dibatalkan pengguna, hentikan program interaktif keseluruhan
        if batal:
            print("\nTerima kasih telah menggunakan Sistem Klasifikasi Kelayakan Air Minum!")
            break
            
        try:
            # Normalisasi input pengguna menggunakan X_min dan range_values dari Data Latih
            user_data_norm = (np.array(user_data) - X_min) / range_values
            
            # Prediksi menggunakan Euclidean Distance dan tetangga terdekat
            jarak = euclidean_distance(X_train, user_data_norm)
            nearest_labels, nearest_indices = get_neighbors(jarak, y_train, best_k)
            
            # Hitung jumlah voting dan persentasenya
            votes_layak = np.sum(nearest_labels == 1)
            votes_tidak_layak = np.sum(nearest_labels == 0)
            
            perc_layak = (votes_layak / best_k) * 100
            perc_tidak_layak = (votes_tidak_layak / best_k) * 100
            
            prediksi = 1 if votes_layak > votes_tidak_layak else 0

            # Tampilkan Hasil Analisis sesuai template laporan
            print("\n" + "="*75)
            print("HASIL ANALISIS KELAYAKAN AIR")
            print("="*75)
            
            print(f"\nNilai K yang digunakan : {best_k}\n")
            
            print("Hasil Voting KNN:")
            print(f"Layak Minum       : {votes_layak}")
            print(f"Tidak Layak Minum : {votes_tidak_layak}\n")
            
            print("Persentase Voting:")
            print(f"Layak Minum       : {perc_layak:.2f}%")
            print(f"Tidak Layak Minum : {perc_tidak_layak:.2f}%\n")

            print("STATUS AIR:")
            if prediksi == 1:
                print("LAYAK MINUM (Potable)\n")
                print("REKOMENDASI PENGGUNAAN AIR:")
                print("1. Minum")
                print("2. Memasak")
            else:
                print("TIDAK LAYAK MINUM (Not Potable)\n")
                print("REKOMENDASI PENGGUNAAN AIR:")
                print("1. Mandi")
                print("2. Mencuci")
            print("="*75)
            
            lanjut = input("\nApakah Anda ingin mengecek sampel air lagi? (y/n): ")
            if lanjut.lower() != 'y':
                print("\nTerima kasih telah menggunakan Sistem Klasifikasi Kelayakan Air Minum!")
                break
            
        except Exception as e:
            print(f"\n[ERROR] Terjadi kesalahan dalam proses kalkulasi: {e}\n")

if __name__ == "__main__":
    main()