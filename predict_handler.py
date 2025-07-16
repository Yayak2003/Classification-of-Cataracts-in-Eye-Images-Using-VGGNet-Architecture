import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from PIL import Image

class PredictHandler:
    def __init__(self, model_path, dataset_path, model_type='vgg16'):
        """
        Initialize the prediction handler.
        
        Args:
            model_path (str): Path to the saved model file.
            dataset_path (str): Path to dataset directory (needed for class mapping).
            model_type (str): Model type ('vgg16' or 'vgg19') to apply the correct preprocessing.
        """
        self.model = load_model(model_path)
        self.img_size = (224, 224)

        # Pilih fungsi preprocessing yang sesuai
        self.preprocess_func = vgg16_preprocess if model_type == 'vgg16' else vgg19_preprocess

        # Ambil label mapping otomatis dari dataset
        self.label_mapping = self.get_label_mapping(dataset_path)
        print("Label Mapping:", self.label_mapping)

    def get_label_mapping(self, dataset_path):
        """
        Get class label mapping from dataset directory.
        
        Args:
            dataset_path (str): Path to dataset directory.
        
        Returns:
            dict: Mapping class index ke nama kelas berdasarkan training dataset.
        """
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            dataset_path,
            target_size=self.img_size,
            batch_size=1,
            class_mode='categorical',
            shuffle=False
        )
        # Ambil label mapping berdasarkan indeks (sesuai training)
        return {v: k for k, v in test_generator.class_indices.items()}

    def preprocess_image(self, image):
        """
        Preprocess image for model prediction.
        
        Args:
            image: PIL Image object or path to image.
        
        Returns:
            Preprocessed image array.
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"File gambar tidak ditemukan: {image}")
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Gambar harus berupa objek Gambar PIL atau path ke gambar")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize image
        img = image.resize(self.img_size)
        
        # Convert to array dan preprocessing sesuai training
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = self.preprocess_func(img_array)  # Preprocessing VGG

        return img_array
        
    def predict(self, image):
        """
        Make prediction on an image.
        
        Args:
            image: PIL Image object or path to image.
        
        Returns:
            tuple: (predicted class, probabilities dict).
        """
        try:
            img_array = self.preprocess_image(image)
            predictions = self.model.predict(img_array, verbose=0)

            # Ambil indeks prediksi tertinggi
            predicted_index = np.argmax(predictions[0])

            # Konversi indeks ke nama kelas menggunakan label_mapping
            predicted_class = self.label_mapping[predicted_index]
            probabilities = {self.label_mapping[i]: float(prob) for i, prob in enumerate(predictions[0])}

            return predicted_class, probabilities
        except Exception as e:
            raise Exception(f"Prediksi gagal: {str(e)}")

# Example usage
if __name__ == '__main__':
    # Pilih model yang telah dilatih sebelumnya
    model_path = 'hasil_model/VGG16/vgg16_lr_0.1_bs_16_epochs_100.h5'  # Ganti dengan path model yang sesuai
    dataset_path = 'dataset/testing'  # Path dataset testing yang digunakan saat training
    image_path = 'dataset/testing/Glaucoma/frg8.png'  # Ganti dengan path gambar uji

    predictor = PredictHandler(model_path, dataset_path, model_type='vgg16')  # Gunakan 'vgg19' jika model yang digunakan adalah VGG19
    
    try:
        pred_class, probs = predictor.predict(image_path)
        print(f"Kelas yang Diprediksi: {pred_class}")
        print("Probabilitas:", probs)
    except Exception as e:
        print(f"Error: {str(e)}")