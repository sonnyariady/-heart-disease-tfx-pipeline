# Proyek Pengembangan Machine Learning Pipeline (Dicoding Submission 1)

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
| **Arsitektur Model** | Deep Neural Network (DNN) berbasis Keras dengan **DenseFeatures**, **Embedding Layers** untuk fitur kategorikal, **Batch Normalization**, 2 Hidden Dense Layers (**Layer 1: 64 unit**, **Layer 2: 64 unit**), **Dropout (0.5)**, **Learning Rate (0.01)** hasil optimasi komponen **Tuner (`tuner.py`)**, dan Sigmoid Output Layer untuk klasifikasi biner. |
| **Metrik Evaluasi** | **Binary Accuracy** (ambang batas `lower_bound` > 0.70), **AUC**, **Precision**, **Recall**, serta **Slicing Analysis** berdasarkan atribut gender (`sex`). |
| **Performa Model** | Hasil komponen **Evaluator**: <br>- **BinaryAccuracy**: **0.9951** (99.51%) <br>- **AUC**: **0.9926** (99.26%) <br>- **Precision**: **0.9951** (99.51%) <br>- **Recall**: **0.9901** (99.01%) <br>Model dinyatakan **Blessed** oleh komponen Evaluator karena telah melampaui batas ambang evaluasi (`lower_bound` 0.70). |

---

## Struktur Berkas Submission

```text
MachineLearningPipeline/
├── sonnyariady-pipeline/        # Folder output artifact seluruh komponen TFX pipeline
├── serving_model/              # Direktori berisi SavedModel hasil ekspor Pusher
├── data/
│   └── heart.csv              # Dataset Heart Disease
├── transform.py                # Module preprocessing (TFX Transform)
├── tuner.py                    # Module hyperparameter tuning (TFX Tuner)
├── trainer.py                  # Module model training & serving signature (TFX Trainer)
├── sonnyariady-submission.ipynb# Notebook Jupyter TFX Pipeline (InteractiveContext)
├── sonnyariady-testing.ipynb   # Notebook pengujian & prediction request TF Serving
├── format-dokumentasi.md       # Berkas dokumentasi proyek (Dicoding Template)
├── README.md                   # Berkas README utama
├── requirements.txt            # Dependency daftar pustaka Python
└── Dockerfile                  # Docker configuration untuk TensorFlow Serving
```
