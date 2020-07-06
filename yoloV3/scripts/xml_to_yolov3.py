import xml.etree.ElementTree as ET
import os
import glob

foldername = os.path.basename(os.getcwd())
if foldername == "scripts": os.chdir("..")


data_dir = '/OIDv4_ToolKit/OID/Dataset/'
Dataset_names_path = "data/names.txt"
Dataset_train = "data/train.txt"
Dataset_test = "data/test.txt"
is_subfolder = True

Dataset_names = []
      
def ParseXML(img_folder, file):
    for xml_file in glob.glob(img_folder+'/*.xml'):
        tree=ET.parse(open(xml_file))
        root = tree.getroot()
        image_name = root.find('filename').text
        img_path = img_folder+'/'+image_name
        for i, obj in enumerate(root.iter('object')):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in Dataset_names:
                Dataset_names.append(cls)
            cls_id = Dataset_names.index(cls)
            xmlbox = obj.find('bndbox')
            OBJECT = (str(int(float(xmlbox.find('xmin').text)))+','
                      +str(int(float(xmlbox.find('ymin').text)))+','
                      +str(int(float(xmlbox.find('xmax').text)))+','
                      +str(int(float(xmlbox.find('ymax').text)))+','
                      +str(cls_id))
            img_path += ' '+OBJECT
        print(img_path)
        file.write(img_path+'\n')

def run_XML_to_YOLOv3():
    for i, folder in enumerate(['train']):
        with open([Dataset_train,Dataset_test][i], "w") as file:
            print(os.getcwd()+data_dir+folder)
            img_path = os.path.join(os.getcwd()+data_dir+folder)
            if is_subfolder:
                for directory in os.listdir(img_path):
                    xml_path = os.path.join(img_path, directory)
                    ParseXML(xml_path, file)
            else:
                ParseXML(img_path, file)

    print("Dataset_names:", Dataset_names)
    with open(Dataset_names_path, "w") as file:
        for name in Dataset_names:
            file.write(str(name)+'\n')

run_XML_to_YOLOv3()
