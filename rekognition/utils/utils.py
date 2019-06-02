import numpy as np

def extract_boxes(image, boxes):
    height = image.shape[0]
    width = image.shape[1]

    box_images = []

    for box in boxes:
        for i in range(len(box)):
            if box[i] < 0:
                box[i] = 0

        # Normalized coordinates
        if is_normalized(box):
            p1 = (int(box[0] * height), int(box[1] * width))
            p2 = (int(box[2] * height), int(box[3] * width))
        else:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))

        box_images.append(image[p1[0]:p2[0], p1[1]:p2[1]])

    return box_images

def is_normalized(box):
    if (box < 1).all():
        return True
    return False