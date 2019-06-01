def extract_boxes(image, boxes, normalized_coordinates = True):
    height = image.shape[0]
    width = image.shape[1]

    box_images = []

    for box in boxes:
        p1 = (int(box[0] * height), int(box[1] * width))
        p2 = (int(box[2] * height), int(box[3] * width))

        box_images.append(image[p1[0]:p2[0], p1[1]:p2[1]])

    return box_images