import xml.etree.ElementTree as ET


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for member in root.findall('size'):
        image_width = float(member[0].text)
        image_height = float(member[1].text)

    bbox_coordinates = []
    for member in root.findall('object'):
        class_name = member[0].text  # class name

        if class_name == 'car':
            class_id = 0
        elif class_name == 'motorcycle':
            class_id = 1
        elif class_name == 'bike':
            class_id = 2
        elif class_name == 'police':
            class_id = 3
        elif class_name == 'ambulance':
            class_id = 4
        elif class_name == 'firetruck':
            class_id = 5
        else:
            class_id = 6 # truck


        # bbox coordinates
        xmin = float(member[4][0].text)
        ymin = float(member[4][1].text)
        xmax = float(member[4][2].text)
        ymax = float(member[4][3].text)
        # store data in list

        width = xmax - xmin
        height = ymax - ymin
        x_center = xmin + width/2
        y_center = ymin + height/2

        norm_width = width / image_width
        norm_height = height / image_height
        norm_x_center = x_center / image_width
        norm_y_center = y_center / image_height

        bbox_coordinates.append([class_id, norm_x_center, norm_y_center, norm_width, norm_height])

    return bbox_coordinates