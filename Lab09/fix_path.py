import xml.etree.ElementTree as ET

num_photo = 482

for i in range(num_photo):

	if i==0 or i==5 or i==6 or i==327:
		continue
    filename = str(i)+'.xml'

    try:
        tree = ET.parse(filename)
        root = tree.getroot()

        for path in root.iter('path'):
            new_path = '/home/0613316/darkflow-master/darkflow-master/melody_img//' + str(i) +'.jpg'
            path.text = new_path
            # tree.write(filename)
            tree.write('new_'+filename)
    except:
        print('no file ' + filename)
