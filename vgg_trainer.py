import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess


class VGGTrainer:
    def __init__(self, train_dir, test_dir, img_size=(224, 224), num_classes=3):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_width, self.img_height = img_size
        self.num_classes = num_classes
        self.train_generator = None
        self.test_generator = None
        self.model = None
        self.history = None

    def setup_data_generators(self, model_type):
        preprocess_func = vgg16_preprocess if model_type == 'vgg16' else vgg19_preprocess

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_func,
            rotation_range=180,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            brightness_range=[0.8, 1.2]
        )

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

    def build_model(self, model_type, learning_rate):
        base_model = VGG16 if model_type == 'vgg16' else VGG19
        base_model = base_model(include_top=False, weights='imagenet', input_shape=(self.img_height, self.img_width, 3))

        x = Flatten()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=output)

        for layer in base_model.layers:
            layer.trainable = False

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_and_save(self, model_type, epochs, learning_rate, batch_size, model_save_dir):
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        self.batch_size = batch_size
        self.setup_data_generators(model_type)

        model_filename = f'{model_type}_lr_{learning_rate}_bs_{batch_size}_epochs_{epochs}.h5'
        model_path = os.path.join(model_save_dir, model_filename)

        callbacks = [
            ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ]

        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.test_generator,
            callbacks=callbacks
        )
        return model_filename

    def evaluate_model(self):
        test_labels = self.test_generator.classes
        predictions = self.model.predict(self.test_generator)
        predicted_classes = np.argmax(predictions, axis=1)

        # Confusion Matrix
        cm = confusion_matrix(test_labels, predicted_classes)
        class_labels = list(self.test_generator.class_indices.keys())

        # Calculate metrics
        metrics = self.calculate_metrics(cm, class_labels)

        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Label Prediksi")
        plt.ylabel("Label Asli")
        plt.title("Confusion Matrix")
        plt.show()

        # Plot training history
        self.plot_training_history()

        return metrics

    @staticmethod
    def calculate_metrics(cm, class_labels):
        num_classes = cm.shape[0]
        total = np.sum(cm)
        metrics = {
            'accuracy': [],
            'sensitivity': [],
            'specificity': [],
            'precision': [],
            'f1_score': []
        }

        for i in range(num_classes):
            TP = cm[i, i]
            FN = np.sum(cm[i, :]) - TP
            FP = np.sum(cm[:, i]) - TP
            TN = total - (TP + FN + FP)

            accuracy = (TP + TN) / total
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            metrics['accuracy'].append(accuracy)
            metrics['sensitivity'].append(recall)
            metrics['specificity'].append(specificity)
            metrics['precision'].append(precision)
            metrics['f1_score'].append(f1)

        # Add average metrics
        metrics['average'] = {
            'accuracy': np.mean(metrics['accuracy']),
            'sensitivity': np.mean(metrics['sensitivity']),
            'specificity': np.mean(metrics['specificity']),
            'precision': np.mean(metrics['precision']),
            'f1_score': np.mean(metrics['f1_score'])
        }

        print("Confusion Matrix and Metrics:")
        print("Overall Metrics:")
        for key, value in metrics['average'].items():
            print(f"{key.capitalize()}: {value:.4f}")

        return metrics

    def plot_training_history(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        # Plot akurasi
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, acc, 'b', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

        # Plot loss
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    train_dir = 'dataset/training'
    test_dir = 'dataset/testing'
    model_save_dir = 'hasil_model'

    configurations = [
        {'model_type': 'vgg16', 'epochs': 50, 'learning_rate': 0.1, 'batch_size': 16},
        {'model_type': 'vgg16', 'epochs': 50, 'learning_rate': 0.01, 'batch_size': 16},
        {'model_type': 'vgg16', 'epochs': 50, 'learning_rate': 0.001, 'batch_size': 16},
        {'model_type': 'vgg16', 'epochs': 50, 'learning_rate': 0.0001, 'batch_size': 16},
        {'model_type': 'vgg16', 'epochs': 100, 'learning_rate': 0.1, 'batch_size': 16},
        {'model_type': 'vgg16', 'epochs': 150, 'learning_rate': 0.1, 'batch_size': 32},
        {'model_type': 'vgg16', 'epochs': 200, 'learning_rate': 0.1, 'batch_size': 64},
        {'model_type': 'vgg19', 'epochs': 50, 'learning_rate': 0.1, 'batch_size': 16},
        {'model_type': 'vgg19', 'epochs': 50, 'learning_rate': 0.01, 'batch_size': 16},
        {'model_type': 'vgg19', 'epochs': 50, 'learning_rate': 0.001, 'batch_size': 16},
        {'model_type': 'vgg19', 'epochs': 50, 'learning_rate': 0.0001, 'batch_size': 16},
        {'model_type': 'vgg19', 'epochs': 100, 'learning_rate': 0.001, 'batch_size': 16},
        {'model_type': 'vgg19', 'epochs': 150, 'learning_rate': 0.001, 'batch_size': 32},
        {'model_type': 'vgg19', 'epochs': 200, 'learning_rate': 0.001, 'batch_size': 64},
    ]

    trainer = VGGTrainer(train_dir, test_dir)

    for config in configurations:
        print(f"Konfigurasi Pelatihan: {config}")
        trainer.build_model(config['model_type'], config['learning_rate'])

        # Menampilkan arsitektur model
        print("\nArsitektur Model:")
        trainer.model.summary()

        model_filename = trainer.train_and_save(
            config['model_type'],
            config['epochs'],
            config['learning_rate'],
            config['batch_size'],
            model_save_dir
        )
        print(f"Model disimpan: {model_filename}")
        metrics = trainer.evaluate_model()
        print(metrics)