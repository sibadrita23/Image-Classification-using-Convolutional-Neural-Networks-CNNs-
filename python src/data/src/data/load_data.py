import tensorflow as tf

def load_data(data_dir):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_data = datagen.flow_from_directory(data_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
    return train_data
