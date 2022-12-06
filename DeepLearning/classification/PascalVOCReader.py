import xml.etree.ElementTree as ET


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    bbox_coordinates = []
    for member in root.findall('object'):
        class_name = member[0].text  # class name

        """
        if (class_name == "Unknown"):
            ...
            als er geen object op image is, leeg label
        """

        # bbox coordinates
        xmin = int(member[4][0].text)
        ymin = int(member[4][1].text)
        xmax = int(member[4][2].text)
        ymax = int(member[4][3].text)
        # store data in list
        bbox_coordinates.append([class_name, xmin, ymin, xmax, ymax])

    return bbox_coordinates