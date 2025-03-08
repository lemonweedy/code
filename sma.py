import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ Buat Data Dummy (Bukan Random)
data = {
    "nilai_mtk": [85, 65, 90, 50, 75, 60, 80, 45, 70, 95, 55, 85, 75, 40, 65],
    "nilai_science": [80, 70, 85, 55, 75, 65, 90, 50, 75, 90, 60, 80, 70, 45, 60],
    "nilai_bhs_indonesia": [75, 65, 80, 60, 70, 60, 85, 55, 70, 85, 65, 75, 70, 50, 65],
    "kehadiran": [95, 75, 90, 60, 85, 70, 95, 65, 80, 90, 70, 85, 80, 60, 75],
    "nilai_bhs_inggris": [80, 60, 85, 50, 70, 65, 90, 45, 65, 85, 55, 75, 70, 40, 60],
    "keaktifan_organisasi": [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],  # 1 = Aktif, 0 = Tidak Aktif
    "jam_belajar": [8, 3, 7, 2, 5, 4, 9, 2, 6, 8, 3, 6, 4, 1, 3],
    "kelulusan": [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0]  # Target: 1 = Lulus, 0 = Tidak Lulus
}

# 2️⃣ Buat DataFrame
df = pd.DataFrame(data)

# 3️⃣ Pisahkan fitur (X) dan target (y)
X = df.drop(columns=["kelulusan"])
y = df["kelulusan"]

# 4️⃣ Bagi dataset menjadi train & test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Standarisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ Buat Model ML Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7️⃣ Prediksi data test
y_pred = model.predict(X_test_scaled)

# 8️⃣ Evaluasi Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Menampilkan pentingnya fitur
feature_importance = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)

# 9️⃣ Function untuk Prediksi dari Input User
def prediksi_kelulusan(nilai_mtk, nilai_science, nilai_bhs_indonesia, kehadiran, nilai_bhs_inggris, keaktifan_organisasi, jam_belajar):
    # Buat DataFrame untuk 1 siswa
    input_data = pd.DataFrame([[nilai_mtk, nilai_science, nilai_bhs_indonesia, kehadiran, nilai_bhs_inggris, keaktifan_organisasi, jam_belajar]],
                              columns=["nilai_mtk", "nilai_science", "nilai_bhs_indonesia", "kehadiran", "nilai_bhs_inggris", "keaktifan_organisasi", "jam_belajar"])
    
    # Standarisasi data input
    input_scaled = scaler.transform(input_data)
    
    # Prediksi dengan model
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)
    
    # Interpretasi hasil
    hasil = "🎓 Diprediksi LULUS SMA" if prediction[0] == 1 else "⚠️ Diprediksi TIDAK LULUS SMA"
    
    return hasil, proba[0]

# 🔟 Contoh Input Manual untuk Prediksi
print("\n==== PREDIKSI KELULUSAN SMA ====")
nilai_mtk = int(input("Masukkan nilai matematika (0-100): "))
nilai_science = int(input("Masukkan nilai science (0-100): "))
nilai_bhs_indonesia = int(input("Masukkan nilai bahasa Indonesia (0-100): "))
kehadiran = int(input("Masukkan persentase kehadiran (0-100): "))
nilai_bhs_inggris = int(input("Masukkan nilai bahasa Inggris (0-100): "))
keaktifan_organisasi = int(input("Apakah aktif organisasi? (1 = Ya, 0 = Tidak): "))
jam_belajar = int(input("Masukkan jam belajar per hari (0-24): "))

# Panggil fungsi prediksi
hasil_prediksi, probabilitas = prediksi_kelulusan(nilai_mtk, nilai_science, nilai_bhs_indonesia, kehadiran, nilai_bhs_inggris, keaktifan_organisasi, jam_belajar)
print(f"\n🔍 Hasil Prediksi: {hasil_prediksi}")
print(f"📊 Probabilitas: Tidak Lulus = {probabilitas[0]:.2f}, Lulus = {probabilitas[1]:.2f}")