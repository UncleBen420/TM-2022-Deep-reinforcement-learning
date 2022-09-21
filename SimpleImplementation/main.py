# This is a sample Python script.
from PIL import Image
from matplotlib import pyplot as plt

from FRCNN.trainfcnn import train, predict, draw_boxes

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


if __name__ == '__main__':
    model, history = train("/media/remy/LaCie/TM/Dota/FRCNN/train", "/media/remy/LaCie/TM/Dota/FRCNN/test", "/media/remy/LaCie/TM/Dota/FRCNN/weights", 20)
    image = Image.open("/media/remy/LaCie/TM/Dota/FRCNN/test/images/P2059_9.png")
    bboxes = predict(image, model, 0.5)
    image = draw_boxes(bboxes, image)
    print(bboxes)

    for key, loss in history.items:
        plt.plot(loss, label=key)
    plt.show()

    plt.imshow(image)
    plt.show()
