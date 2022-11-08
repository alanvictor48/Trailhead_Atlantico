import matplotlib.pyplot as plt
from skimage import data

def load_image(file: str):
    return plt.imread(str)

def save_image(image, name: str = 'saved_image'):
    plt.imsave(name, image)

def show_image(image, title: str = 'image'):
    plt.imshow(image)
    plt.title(title)
    plt.show()

def load_coffe_image():
    return data.coffe()

def load_astronaut_image():
    return data.astronaut()