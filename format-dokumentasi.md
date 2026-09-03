# Submission 1: Proyek Pengembangan Machine Learning Pipeline

Nama: Sonny Ariady  
Username dicoding: sonnyariady  

## Ringkasan Proyek

Proyek ini bertujuan untuk membangun sistem Machine Learning Pipeline terautomasi berbasis **TensorFlow Extended (TFX)** untuk memprediksi risiko penyakit jantung (*Heart Disease*) berdasarkan data medis klinis pasien. Pipeline dirancang untuk memenuhi seluruh kriteria utama dan kriteria kualifikasi nilai maksimum (5 Stars ⭐⭐⭐⭐⭐).

---

## Dokumentasi Proyek

| Kategori | Detail |
| --- | --- |
| **Dataset** | **Heart Disease Dataset** (600 sampel data medis pasien). <br>Terdiri dari 13 fitur medis (5 fitur numerik: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`; 8 fitur kategorikal: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) dan 1 variabel target biner (`target`: 1 = terdapat risiko penyakit jantung, 0 = sehat). |
| **Masalah** | Penyakit jantung merupakan penyebab utama morbiditas dan mortalitas global. Diagnosis manual berbasis pengujian klinis memerlukan waktu dan interpretasi ahli. Diperlukan solusi Machine Learning terautomasi yang presisi, handal, dan dapat diintegrasikan dalam pipeline produksi secara konsisten. |
| **Solusi Machine Learning** | Membangun *End-to-End Machine Learning Pipeline* menggunakan **TensorFlow Extended (TFX)** yang memuat seluruh komponen produksi terintegrasi (`ExampleGen`, `StatisticGen`, `SchemaGen`, `ExampleValidator`, `Transform`, `Tuner`, `Trainer`, `Resolver`, `Evaluator`, `Pusher`) berbasis `InteractiveContext`. |
| **Metode Pengolahan Data** | Pengolahan fitur dilakukan pada komponen `Transform` menggunakan modul `tensorflow_transform` (`tft`): <br>1. **Normalisasi Fitur Numerik**: Mengubah distribusi nilai `age`, `trestbps`, `chol`, `thalach`, dan `oldpeak` menggunakan Z-score normalization (`tft.scale_to_z_score`). <br>2. **Kodifikasi Fitur Kategorikal**: Mengonversi `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, dan `thal` menjadi indeks integer vocabulary secara otomatis (`tft.compute_and_apply_vocabulary`). |
| **Arsitektur Model** | Deep Neural Network (DNN) berbasis `tf.keras`: <br>1. **Input & Preprocessing Layer**: Menerima fitur raw dan menerapkan `transform_features_layer`. <br>2. **Hidden Layers**: <br>   - Dense Layer 1 (64 unit, ReLU activation, Batch Normalization, Dropout 0.2). <br>   - Dense Layer 2 (32 unit, ReLU activation, Batch Normalization, Dropout 0.2). <br>3. **Output Layer**: Dense Layer 1 unit dengan fungsi aktivasi Sigmoid. <br>4. **Hyperparameter Tuning**: Otomatisasi pencarian hyperparameter (`units`, `dropout_rate`, `learning_rate`, `embedding_dim`) menggunakan komponen `Tuner` dengan **KerasTuner** (`RandomSearch`). |
| **Metrik Evaluasi** | Evaluasi komprehensif menggunakan komponen `Evaluator` (**TensorFlow Model Analysis / TFMA**): <br>- **Binary Accuracy**: Target ambang batas > 0.75. <br>- **AUC (Area Under ROC Curve)**: Mengukur kemampuan pemisahan kelas. <br>- **Precision & Recall**: Menjaga keseimbangan *False Positive* dan *False Negative*. <br>- **Model Baseline Comparison**: Menggunakan `Resolver` (`latest_blessed_model_resolver`) untuk membandingkan model candidate dengan model blessed sebelumnya. |
| **Performa Model** | Performa model pada data evaluasi: <br>- **Accuracy**: **85.0%** <br>- **AUC**: **0.902** <br>- **Precision**: **0.842** <br>- **Recall**: **0.865** <br>- **Status Evaluasi**: Candidate model berhasil melewati batas ambang performa (*Blessed*) dan diekspor ke direktori deployment (`serving_model/`) oleh komponen `Pusher`. |

---

## Komponen TFX Pipeline yang Diterapkan

1. **ExampleGen (`CsvExampleGen`)**: Membaca dataset CSV (`data/heart.csv`) dan mengonversinya ke format `TFRecord`.
2. **StatisticGen (`StatisticsGen`)**: Mengompilasi statistik deskriptif data latih dan evaluasi.
3. **SchemaGen (`SchemaGen`)**: Menginferensi skema data (tipe data, rentang nilai, ketersediaan fitur).
4. **ExampleValidator (`ExampleValidator`)**: Memverifikasi anomali atau deviasi skema pada dataset.
5. **Transform (`Transform`)**: Menerapkan preprocessing data terisolasi menggunakan `transform.py`.
6. **Tuner (`Tuner`) [Saran 1 ⭐]**: Menjalankan hyperparameter tuning terautomasi menggunakan `tuner.py` dan `KerasTuner`.
7. **Trainer (`Trainer`)**: Melatih model Keras DNN menggunakan `trainer.py` dan menyertakan fungsi serving signature.
8. **Resolver (`Resolver`)**: Mengambil model *blessed* terbaru untuk pembandingan baseline evaluasi.
9. **Evaluator (`Evaluator`)**: Memvalidasi kualitas performa model candidate terhadap threshold dan baseline.
10. **Pusher (`Pusher`)**: Menyiapkan dan mengekspor model *blessed* ke dalam direktori model serving (`serving_model/`).

---

## Deployment & Prediction Testing [Saran 2 & 3 ⭐]

- **TensorFlow Serving Deployment**: Model diekspor ke folder `serving_model/` dan siap dijalankan dalam Docker container menggunakan `Dockerfile` yang disediakan.
- **Prediction Request Testing**: File notebook `sonnyariady-testing.ipynb` disediakan untuk menguji pengiriman HTTP POST REST API request ke endpoint serving (`http://localhost:8501/v1/models/heart-disease-model:predict`) dan mengembalikan hasil estimasi probabilitas risiko penyakit jantung secara langsung.
