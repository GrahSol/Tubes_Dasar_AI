import pandas as pd
import numpy as np

# ==============================================================================
# TUGAS BESAR KECERDASAN BUATAN
# Sistem Klasifikasi Kelayakan Air Minum Menggunakan KNN
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. FUNGSI KNN (K-Nearest Neighbor)
# ------------------------------------------------------------------------------
def euclidean_distance(X_train, test_row):
    """
    Menghitung jarak Euclidean (L2) menggunakan optimasi broadcasting NumPy.
    """
    distances = np.sqrt(np.sum((X_train - test_row) ** 2, axis=1))
    return distances

def get_neighbors(distances, y_train, k):
    """
    Mencari K tetangga dengan jarak terdekat.
    """
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]
    return nearest_labels, nearest_indices

def majority_voting(nearest_labels):
    """
    Melakukan pemungutan suara (voting) untuk menentukan prediksi akhir.
    """
    labels_list = nearest_labels.tolist()
    prediction = max(set(labels_list), key=labels_list.count)
    return prediction

def predict_knn(X_train, y_train, X_test, k):
    """
    Menjalankan alur prediksi untuk seluruh dataset uji secara iteratif (Lazy Learning).
    """
    predictions = []
    for test_row in X_test:
        distances = euclidean_distance(X_train, test_row)
        nearest_labels, _ = get_neighbors(distances, y_train, k)
        pred = majority_voting(nearest_labels)
        predictions.append(pred)
    return np.array(predictions)

# ------------------------------------------------------------------------------
# 2. FUNGSI EVALUASI MODEL 
# ------------------------------------------------------------------------------
def evaluate_model(y_true, y_pred):
    """
    Menghitung metrik performa model dasar dan Confusion Matrix.
    - True Positive (TP) : Asli Layak (1), Prediksi Layak (1)
    - True Negative (TN) : Asli Tidak Layak (0), Prediksi Tidak Layak (0)
    - False Positive (FP): Asli Tidak Layak (0), Prediksi Layak (1) -> Error Fatal
    - False Negative (FN): Asli Layak (1), Prediksi Tidak Layak (0) -> Error Aman
    """
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
    print("="*72)
    print("SISTEM KLASIFIKASI KELAYAKAN AIR MINUM")
    print("MENGGUNAKAN ALGORITMA K-NEAREST NEIGHBOR (KNN)")
    print("="*72)

    print("\n[1] Membaca Dataset dan Preprocessing...")
    try:
        df = pd.read_csv('water_potability.csv')
    except FileNotFoundError:
        print("ERROR: File 'water_potability.csv' tidak ditemukan!")
        return

    missing_before = df.isnull().sum().sum()
    print(f"    -> Ditemukan total {missing_before} missing value.")
    
    df.fillna(df.mean(), inplace=True)
    
    duplikat = df.duplicated().sum()
    print(f"    -> Ditemukan {duplikat} data duplikat.")
    if duplikat > 0:
        df.drop_duplicates(inplace=True)

    X = df.drop('Potability', axis=1).values
    y = df['Potability'].values
    fitur_names = df.drop('Potability', axis=1).columns.tolist()

    aturan_skala = {
        'ph':              (0.0, 14.0, "0.0 - 14.0"),
        'hardness':        (0.0, 350.0, "0.0 - 350.0 mg/L"),
        'solids':          (0.0, 65000.0, "0.0 - 65000.0 ppm"),
        'chloramines':     (0.0, 15.0, "0.0 - 15.0 ppm"),
        'sulfate':         (0.0, 500.0, "0.0 - 500.0 mg/L"),
        'conductivity':    (0.0, 800.0, "0.0 - 800.0 µS/cm"),
        'organic_carbon':  (0.0, 35.0, "0.0 - 35.0 ppm"),
        'trihalomethanes': (0.0, 130.0, "0.0 - 130.0 µg/L"),
        'turbidity':       (0.0, 8.0, "0.0 - 8.0 NTU")
    }

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.8)
    
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    X_train_raw, X_test_raw = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    X_min = X_train_raw.min(axis=0)
    X_max = X_train_raw.max(axis=0)
    
    range_values = X_max - X_min
    range_values[range_values == 0] = 1
    
    X_train = (X_train_raw - X_min) / range_values
    X_test = (X_test_raw - X_min) / range_values

    print(f"    -> Normalisasi selesai. Data Latih: {len(X_train)} baris, Data Uji: {len(X_test)} baris.")

    print("\n[2] Mengevaluasi Hyperparameter K (Fase Pelatihan & Pengujian)...")
    
    # mencari nilai k terbaik
    k_values = [1, 3, 5, 7, 9]
    best_k = -1
    best_acc = 0.0
    best_f1 = 0.0
    best_cm = (0, 0, 0, 0)

    garis_tabel = "-" * 72
    print(garis_tabel)
    print(f"| {'K':<3} | {'Akurasi':<8} | {'TP':<4} | {'TN':<4} | {'FP (Fatal)':<10} | {'FN':<4} | {'F1-Score':<8} |")
    print(garis_tabel)

    for k in k_values:
        y_pred = predict_knn(X_train, y_train, X_test, k)
        acc, tp, tn, fp, fn = evaluate_model(y_test, y_pred)
        
        # Kalkulasi F1-Score untuk dicantumkan sebagai metrik pendukung
        f1_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        print(f"| {k:<3} | {acc*100:>7.2f}% | {tp:<4} | {tn:<4} | {fp:<10} | {fn:<4} | {f1_score*100:>7.2f}% |")
        
        # KEMBALI KE RENCANA UTAMA: Mengunci K terbaik mutlak berdasarkan ACCURACY tertinggi
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_f1 = f1_score
            best_cm = (tp, tn, fp, fn)

    print(garis_tabel)
    
    # Menampilkan Ringkasan K Terbaik Berdasarkan Accuracy
    print(f"\n[HASIL TERBAIK] Nilai optimal berdasarkan Accuracy adalah K={best_k}")
    print(f"Metrik Utama Seleksi : Accuracy = {best_acc*100:.2f}%")
    print(f"Metrik Pendukung     : F1-Score = {best_f1*100:.2f}%")
    print("-" * 72)
    print("Detail Confusion Matrix pada K Terbaik:")
    print(f" - True Positive (TP)  : {best_cm[0]:<4} (Benar: Asli Layak, Prediksi Layak)")
    print(f" - True Negative (TN)  : {best_cm[1]:<4} (Benar: Asli Tidak Layak, Prediksi Tidak Layak)")
    print(f" - False Positive (FP) : {best_cm[2]:<4} (FATAL: Asli Tidak Layak, Prediksi Layak)")
    print(f" - False Negative (FN) : {best_cm[3]:<4} (Error: Asli Layak, Prediksi Tidak Layak)")
    print("-" * 72)

    print("\n" + "="*72)
    print("INPUT PENGGUNA UNTUK PREDIKSI AIR BARU")
    print("="*72)

    while True:
        user_data = []
        batal = False 
        
        print("\nSilakan masukkan nilai parameter kualitas air (Ketik 'q' untuk keluar):")
        
        for fitur in fitur_names:
            key_lookup = fitur.lower().replace(" ", "_")
            if key_lookup in aturan_skala:
                min_riil, max_riil, label_skala = aturan_skala[key_lookup]
            else:
                min_riil, max_riil, label_skala = (0.0, 1000000.0, "0 - 1.000.000")
            
            while True:
                val = input(f"Masukkan nilai {fitur} (Skala Distribusi: {label_skala}): ")
                if val.lower() == 'q':
                    batal = True
                    break
                try:
                    val_float = float(val)
                    if not (min_riil <= val_float <= max_riil):
                        print(f"    [VALIDASI] Nilai menembus batas distribusi wajar! {fitur} harus berada di rentang {label_skala}.")
                        continue
                    user_data.append(val_float)
                    break
                except ValueError:
                    print("    [ERROR] Masukan tidak valid! Harus berupa angka.\n")
            
            if batal:
                break
        
        if batal:
            print("\nTerima kasih telah menggunakan Sistem Klasifikasi Kelayakan Air Minum!")
            break
            
        try:
            user_data_norm = (np.array(user_data) - X_min) / range_values
            
            garis_norm = "=" * 108
            print("\n" + garis_norm)
            print("TAHAP 1: PROSES NORMALISASI INPUT PENGGUNA (MIN-MAX SCALING)")
            print(garis_norm)
            print(f"| {'Parameter (Fitur)':<17} | {'Input Asli (X)':<14} | {'X_min Dataset':<13} | {'X_max Dataset':<13} | {'Perhitungan ke Skala 0-1 (X_norm)':<33} |")
            print("-" * 108)
            for i, fitur in enumerate(fitur_names):
                x_val = user_data[i]
                xmin_val = X_min[i]
                xmax_val = X_max[i]
                xnorm_val = user_data_norm[i]
                rumus_norm = f"({x_val:.2f} - {xmin_val:.2f}) / {range_values[i]:.2f} = {xnorm_val:.4f}"
                print(f"| {fitur:<17} | {x_val:<14.2f} | {xmin_val:<13.2f} | {xmax_val:<13.2f} | {rumus_norm:<33} |")
            print(garis_norm)

            jarak = euclidean_distance(X_train, user_data_norm)
            nearest_labels, nearest_indices = get_neighbors(jarak, y_train, best_k)
            
            batas_garis = "=" * 148
            garis_pisah = "-" * 148
            
            print("\n" + batas_garis)
            print("TAHAP 2: URUTKAN DATA BERDASARKAN HASIL PERHITUNGAN JARAK EUCLIDEAN")
            print(batas_garis)
            
            print(f"| {'Baris':<5} | {'pH':<4} | {'Hard':<4} | {'Solid':<5} | {'Chlo':<4} | {'Sulf':<4} | {'Cond':<4} | {'OrgC':<4} | {'Trih':<4} | {'Turb':<4} | {'Status':<11} | {'Jarak Euclidean (Substitusi Rumus)':<44} | {'Rank':<4} |")
            print(garis_pisah)
            
            def fmt(v, w):
                s = f"{v:.1f}"
                if len(s) > w: s = f"{v:.0f}"
                return s[:w].ljust(w)

            for rank, idx in enumerate(nearest_indices):
                label_teks = "Layak" if nearest_labels[rank] == 1 else "Tidak Layak"
                raw = X_train_raw[idx]
                norm = X_train[idx]
                
                u1, u2 = user_data_norm[0], user_data_norm[1]
                t1, t2 = norm[0], norm[1]
                
                rumus_teks = f"(({u1:.2f}-{t1:.2f})² + ({u2:.2f}-{t2:.2f})² + ...) = {jarak[idx]:.4f}"
                
                print(f"| {idx:<5} | {fmt(raw[0],4)} | {fmt(raw[1],4)} | {fmt(raw[2],5)} | {fmt(raw[3],4)} | {fmt(raw[4],4)} | {fmt(raw[5],4)} | {fmt(raw[6],4)} | {fmt(raw[7],4)} | {fmt(raw[8],4)} | {label_teks:<11} | {rumus_teks:<44} | {rank+1:<4} |")
            
            print(garis_pisah)
            
            u = user_data
            print(f"| {'INPUT':<5} | {fmt(u[0],4)} | {fmt(u[1],4)} | {fmt(u[2],5)} | {fmt(u[3],4)} | {fmt(u[4],4)} | {fmt(u[5],4)} | {fmt(u[6],4)} | {fmt(u[7],4)} | {fmt(u[8],4)} | {'?':<11} | {'-':<44} | {'-':<4} |")
            print(batas_garis)

            votes_layak = np.sum(nearest_labels == 1)
            votes_tidak_layak = np.sum(nearest_labels == 0)
            
            perc_layak = (votes_layak / best_k) * 100
            perc_tidak_layak = (votes_tidak_layak / best_k) * 100
            
            prediksi = 1 if votes_layak > votes_tidak_layak else 0

            print("\n" + "="*72)
            print("TAHAP 3: HASIL ANALISIS KELAYAKAN AIR")
            print("="*72)
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
            else:
                print("TIDAK LAYAK MINUM (Not Potable)\n")
            print("="*72)
            
            lanjut = input("\nApakah Anda ingin mengecek sampel air lagi? (y/n): ")
            if lanjut.lower() != 'y':
                print("\nPROGRAM SELESAI")
                break
            
        except Exception as e:
            print(f"\n[ERROR] Terjadi kesalahan dalam proses kalkulasi: {e}\n")

if __name__ == "__main__":
    main()