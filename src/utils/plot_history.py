import matplotlib.pyplot as plt

def plot_history(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.title('Training accuracy and loss')
    plt.legend()
    plt.show()
