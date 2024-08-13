import tensorflow as tf
from src.data.load_data import load_data
from src.models.build_cnn import build_cnn

def train_model(data_dir, epochs=10):
    train_data = load_data(data_dir)
    model = build_cnn()
    history = model.fit(train_data, epochs=epochs)
    model.save('models/image_classifier.h5')
    return history

if __name__ == '__main__':
    history = train_model('data/processed/')
