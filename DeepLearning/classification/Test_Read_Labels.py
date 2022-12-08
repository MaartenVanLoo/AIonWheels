import os

import matplotlib.pyplot as plt

from PascalVOCReader import read_content
from PascalVOCReader import print_content

if __name__ == '__main__':
    files = os.listdir(r"C:\Users\maxim\OneDrive - Universiteit Antwerpen\GitHub\AIonWheels\DeepLearning\classification\data")
    for xml in files:
        if xml.endswith('.xml'):
            bbox_coordinates = read_content("data/"+xml)
            print(bbox_coordinates)

            image_name = xml.replace(".xml", ".png")
            image_path = "data/"+image_name
            bbox_coordinates.insert(0, [image_path])
            print(bbox_coordinates)

            #print_content(bbox_coordinates)

