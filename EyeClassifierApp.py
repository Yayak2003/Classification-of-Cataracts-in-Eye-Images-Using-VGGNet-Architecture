import streamlit as st
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from predict_handler import PredictHandler

class EyeClassifierApp:
    def __init__(self):
        self.model_dir = 'hasil_model'
        self.dataset_path = 'dataset/testing'  # Pastikan path dataset sesuai
        self.predictor = None
        self.selected_architecture = None
        self.selected_model = None
        self.metrics_data = None

        # Mapping nama model ke format "Model V1", "Model V2", dll.
        self.model_mapping = {
            "VGG16": [
                "vgg16_lr_0.1_bs_16_epochs_50.h5",
                "vgg16_lr_0.01_bs_16_epochs_50.h5",
                "vgg16_lr_0.001_bs_16_epochs_50.h5",
                "vgg16_lr_0.0001_bs_16_epochs_50.h5",
                "vgg16_lr_0.1_bs_16_epochs_100.h5",
                "vgg16_lr_0.1_bs_32_epochs_150.h5",
                "vgg16_lr_0.1_bs_64_epochs_200.h5",
            ],
            "VGG19": [
                "vgg19_lr_0.1_bs_16_epochs_50.h5",
                "vgg19_lr_0.01_bs_16_epochs_50.h5",
                "vgg19_lr_0.001_bs_16_epochs_50.h5",
                "vgg19_lr_0.0001_bs_16_epochs_50.h5",
                "vgg19_lr_0.01_bs_16_epochs_100.h5",
                "vgg19_lr_0.01_bs_32_epochs_150.h5",
                "vgg19_lr_0.01_bs_64_epochs_200.h5",
            ]
        }
        
        # Load metrics data
        self.load_metrics_data()

    def load_metrics_data(self):
        try:
            # Muat data metrics dari CSV
            self.metrics_data = pd.read_csv('hasil_metrics.csv', sep=';')
            # Bersihkan data jika diperlukan (misalnya hapus spasi)
            self.metrics_data = self.metrics_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            # Convert numeric columns to float
            for col in ['Akurasi', 'Sensitivitas', 'Spesifisitas', 'Presisi', 'F1-Score']:
                self.metrics_data[col] = self.metrics_data[col].astype(float)
        except Exception as e:
            st.error(f"Error loading metrics data: {e}")
            self.metrics_data = None

    def set_custom_styles(self):
        """Mengubah warna sidebar, tombol, teks, dan efek hover."""
        st.markdown("""
            <style>
                [data-testid="stSidebar"] {
                    background-color: #000033 !important; 
                }
                [data-testid="stExpander"] {
                    background-color: #000033 !important; /* Warna latar belakang expander */
                    color: white !important; /* Warna teks dalam expander */
                    border: 1px solid #00aaff !important; /* Warna border expander */
                }
                [data-testid="stExpander"] div {
                    color: white !important; /* Warna teks dalam konten expander */
                }
                .stButton>button:hover {
                    background-color: #00033EE !important; /* Hover button biru */
                    color: white !important;
                }
                button:hover, .btn:hover {
                background-color: blue !important;
                border-color: blue !important;
                color: white !important;
                }
                label, .config-title {
                    color: white !important; /* Ubah warna teks sidebar */
                }
                .model-config {
                    color: white !important; /* Ubah warna teks konfigurasi model */
                    font-weight: bold;
                }
                .model-loaded-btn {
                    color: white !important; /* Warna teks tombol model dimuat */
                    background-color: rgba(0, 255, 0, 0.15) !important; /* Hijau transparan */
                    box-shadow: 0px 0px 5px white !important; /* Efek shadow */
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                }
                .prediksi-hasil {
                    color: white !important; /* Teks tetap putih */
                    background-color: #006400 !important; /* Hijau tua */
                    box-shadow: 0px 0px 5px #1E90FF !important; /* Efek glow hijau tua */
                    padding: 5px;
                    border-radius: px;
                    text-align: center;
                    font-size: 17px;
                    margin-top: 5px;
                }
                .metrics-value {
                    color: white !important;
                    font-weight: bold;
                    font-size: 16px;
                    padding: 5px;
                    margin: 5px 0;
                    background-color: rgba(0, 50, 100, 0.3);
                    border-radius: 4px;
                }
            </style>
        """, unsafe_allow_html=True)

    def load_model(self):
        try:
            self.selected_architecture = st.sidebar.selectbox("Pilih Arsitektur", ["VGG16", "VGG19"], key='architecture')
            model_files = self.model_mapping[self.selected_architecture]

            # Tentukan base_index sesuai arsitektur
            if self.selected_architecture == "VGG16":
                 base_index = 0  # Model V1 - V7
            else:  # VGG19
                 base_index = 7  # Mulai dari Model V8
                 
            # Buat nama tampilan model
            model_display_names = [f"Model V{base_index + i + 1}" for i in range(len(model_files))]


            selected_model_index = st.sidebar.selectbox("Pilih Model", range(len(model_display_names)), format_func=lambda i: model_display_names[i])
            self.selected_model = model_files[selected_model_index]
            #self.selected_model_index = selected_model_index
            model_path = os.path.join(self.model_dir, self.selected_architecture.lower(), self.selected_model)
            self.predictor = PredictHandler(model_path, self.dataset_path)
            model_config = self.parse_model_filename(self.selected_model)

            # Simpan global model index (untuk ditampilkan sebagai Model Vx)
            self.selected_model_index = base_index + selected_model_index

            st.sidebar.markdown('<h3 class="model-config">Konfigurasi Model</h3>', unsafe_allow_html=True)
            st.sidebar.markdown(f'<p class="model-config">- <strong>Nama Model:</strong> {self.selected_architecture} - V{self.selected_model_index + 1}</p>', unsafe_allow_html=True)
            st.sidebar.markdown(f'<p class="model-config">- <strong>Learning Rate:</strong> {model_config.get("learning_rate", "Unknown")}</p>', unsafe_allow_html=True)
            st.sidebar.markdown(f'<p class="model-config">- <strong>Batch Size:</strong> {model_config.get("batch_size", "Unknown")}</p>', unsafe_allow_html=True)
            st.sidebar.markdown(f'<p class="model-config">- <strong>Epochs:</strong> {model_config.get("epochs", "Unknown")}</p>', unsafe_allow_html=True)

            st.sidebar.markdown(f'<div class="model-loaded-btn"> Dimuat Model <strong>{self.selected_architecture}</strong> : <strong>{model_display_names[selected_model_index]}</strong></div> ', unsafe_allow_html=True)
            return True
        except Exception as e:
            st.error(f"Kesalahan memuat model: {e}")
            return False
    
    def parse_model_filename(self, filename):
        try:
            parts = filename.replace(".h5", "").split("_")
            return {
                "learning_rate": float(parts[2]),
                "batch_size": int(parts[4]),
                "epochs": int(parts[6])
            }
        except (IndexError, ValueError):
            return {"learning_rate": "Unknown", "batch_size": "Unknown", "epochs": "Unknown"}
    
    def get_model_metrics(self, model_index):
        """Mendapatkan metrik evaluasi dari data CSV berdasarkan indeks model."""
        if self.metrics_data is not None:
            row_name = f"Model {model_index + 1}"
            model_metrics = self.metrics_data[self.metrics_data['Model '].str.strip() == row_name]
            if not model_metrics.empty:
                return model_metrics.iloc[0]
        return None

    
    def create_confusion_matrix_heatmap(self, pred_class, probs):
        """Membuat heatmap sebagai visualisasi prediksi."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Kelas dan probabilitas
        classes = list(probs.keys())
        probabilities = [probs[cls] for cls in classes]
        
        # Buat heatmap sederhana untuk representasi visual
        data = [[0] * len(classes) for _ in range(len(classes))]
        pred_idx = classes.index(pred_class)
        
        # Tandai kelas prediksi
        for i, cls in enumerate(classes):
            data[i][i] = probs[cls]
        
        # Buat heatmap
        sns.heatmap(data, annot=True, fmt=".2%", cmap="Blues", 
                   xticklabels=classes, yticklabels=classes, ax=ax)
        plt.title("Heatmap Prediksi")
        plt.xlabel("Kelas Prediksi")
        plt.ylabel("Kelas Sebenarnya (unknown)")
        
        return fig
    
    def run(self):
        st.set_page_config(page_title="Klasifikasi Citra Mata", layout="centered")
        self.set_custom_styles()
        st.title("Klasifikasi Citra Mata dengan Arsitektur VGGNet")
        st.markdown("Aplikasi ini menggunakan model pembelajaran Deep Learning untuk mengklasifikasikan gambar mata ke dalam 3 kategori: Katarak, Glaukoma, dan Normal.")
        
        if self.load_model():
            uploaded_file = st.file_uploader("Unggah Gambar Mata", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col2:
                        pred_button = st.button("Prediksi")
                    if pred_button:
                        with st.spinner("Memprediksi..."):
                            pred_class, probs = self.predictor.predict(image)
                        st.markdown(f'<div class="prediksi-hasil">Kelas yang Diprediksi: <strong>{pred_class}</strong></div>', unsafe_allow_html=True)
                        
                        # Dapatkan metrik dari data CSV
                        model_metrics = self.get_model_metrics(self.selected_model_index)

                        
                        with st.expander("Detail Prediksi"):
                            # Tampilkan probabilitas prediksi
                            st.subheader("Probabilitas Kelas")
                            for cls, prob in probs.items():
                                st.markdown(f"<div class='metrics-value'>{cls}: {prob*100:.2f}%</div>", unsafe_allow_html=True)
                            
                            # Tampilkan metrik evaluasi model
                            if model_metrics is not None:
                                st.subheader("Metrik Evaluasi Model")
                                st.markdown(f"<div class='metrics-value'>Akurasi: {int(model_metrics['Akurasi']*100)}%</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='metrics-value'>Sensitivitas: {int(model_metrics['Sensitivitas']*100)}%</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='metrics-value'>Spesifisitas: {int(model_metrics['Spesifisitas']*100)}%</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='metrics-value'>Presisi: {int(model_metrics['Presisi']*100)}%</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='metrics-value'>F1-Score: {int(model_metrics['F1-Score']*100)}%</div>", unsafe_allow_html=True)
                            else:
                                st.warning("Metrik evaluasi untuk model ini tidak ditemukan!")
                            
                            # Tampilkan confusion matrix jika tersedia
                            cm_filename = f"CM_{self.selected_model.replace('.h5', '.png')}"
                            cm_path = os.path.join(self.model_dir, cm_filename)
                            if os.path.exists(cm_path):
                                st.subheader("Confusion Matrix")
                                st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
                            else:
                                # Buat heatmap prediksi sederhana
                                st.subheader("Heatmap Prediksi")
                                fig = self.create_confusion_matrix_heatmap(pred_class, probs)
                                st.pyplot(fig)
                except Exception as e:
                    st.error(f"Kesalahan memproses gambar: {str(e)}")

if __name__ == '__main__':
    app = EyeClassifierApp()
    app.run()