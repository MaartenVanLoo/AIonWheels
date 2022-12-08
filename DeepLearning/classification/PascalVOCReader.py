import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    print("xml_file:", xml_file)

    bbox_coordinates = []
    for member in root.findall('object'):
        class_name = member[0].text  # class name

        # bbox coordinates
        xmin = float(member[4][0].text)
        ymin = float(member[4][1].text)
        xmax = float(member[4][2].text)
        ymax = float(member[4][3].text)
        # store data in list
        bbox_coordinates.append([class_name, xmin, ymin, xmax, ymax])

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