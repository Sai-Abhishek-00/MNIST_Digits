# load and prepare the image
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model


def load_image(filename):
    # load & convert image
    sample = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert image to array
    img = img_to_array(sample)
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


# load an image and predict the class
def run_example():
    # load the image
    img = load_image('sample_image-300x298.png')
    # load model
    model = load_model('digit_recog_model.h5')
    # predict the class
    digit = model.predict_classes(img)
    print(digit[0])


# entry point, run the example
run_example()