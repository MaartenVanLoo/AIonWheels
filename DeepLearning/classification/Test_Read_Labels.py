import os

from PascalVOCReader import read_content


if __name__ == '__main__':
    os.mkdir("labels")
    files = os.listdir("images")
    for xml in files:
        if xml.endswith('.xml'):
            bbox_coordinates = read_content("images/"+xml)

            path = "labels/"+xml.replace(".xml", ".txt")
            with open(path, 'w') as fp:
                for object in bbox_coordinates:
                    fp.write(str(object[0])+" "+str(object[1])+" "
                    +str(object[2])+" "+str(object[3])+" "+str(object[4])+"\n")

            os.remove('images/'+xml)

