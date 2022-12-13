import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    #print("xml_file:", xml_file)

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


def print_content(info_list: list):
    img = plt.imread(info_list[0][0])
    #text = ""
    #for i in range(1, len(info_list)):
        #text = text + str(info_list[i][0])
        #text = "[{} , {} , {}, {}, {}]".format(info_list[i][0], info_list[i][1], info_list[i][2], info_list[i][3], info_list[i][4])

    """
        if i == 0:
            img = mpimg.imread(info_list[i][0])
            
            print("Image:", info_list[i][0])
        else:
            print("Type:", info_list[i][0],
                  "\nCoordinates:",
                  "\n\txmin:", info_list[i][1],
                  "\n\tymin:", info_list[i][2],
                  "\n\txmax:", info_list[i][3],
                  "\n\tymax:", info_list[i][4],"\n")
                  """

    #plt.text(i*20, 0, text, fontsize=8, bbox=dict(fill=False, edgecolor='white', linewidth=5))
    plt.imshow(img)
    plt.show()
    plt.pause(5)
    plt.close()