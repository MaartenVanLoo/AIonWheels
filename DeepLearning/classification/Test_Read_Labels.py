import os

from PascalVOCReader import read_content
from PascalVOCReader import print_content

if __name__ == '__main__':
    files = os.listdir(r"/DeepLearning/classification/images")
    for xml in files:
        if xml.endswith('.xml'):
            bbox_coordinates = read_content("images/"+xml)
            #print(bbox_coordinates)

            path = "labels/"+xml.replace(".xml", ".txt")
            with open(path, 'w') as fp:
                for object in bbox_coordinates:
                    fp.write(str(object[0])+" "+str(object[1])+" "
                    +str(object[2])+" "+str(object[3])+" "+str(object[4])+"\n")


            os.remove('images/'+xml)

        #we add the image to the list with his labels
        #can also be placed in the read_content function if necessary
        """
        image_name = xml.replace(".xml", ".png")
        image_path = "images/"+image_name
        bbox_coordinates.insert(0, [image_path])
        print(bbox_coordinates)
        """
        #print_content(bbox_coordinates)

