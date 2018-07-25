import SimpleITK as sitk



def load_the_dataset_to_list(data_dir,CaseList):

    print("loading image...")

    file_name = []
    with open(CaseList) as f:
        all_line = f.readlines()
        for line in all_line:
            file_name.append(line.replace("\n","")) 
  
    x = []
    t = []
    for i in range(len(file_name)):
        Image_x = sitk.ReadImage(os.path.join(data_dir,"downsampledCT",file_name[i]+".mhd"))
        Image_t = sitk.ReadImage(os.path.join(data_dir,"downsampledLabel",file_name[i]+".mhd"))

        x.append(Image_x)
        t.append(Image_t)

    return (x,y)


def load_text(filepath):
    file_name = []
    print("load " + filepath ) 
    with open(filepath) as f:
        all_line = f.readlines()
        for line in all_line:
            file_name.append(line.replace("\n",""))  
    return file_name


def load_coord_text(filepath):
    print("load " + filepath ) 
    with open(filepath) as f:
        all_line = f.readlines()
    VV = [[0 for i in range(4)] for j in range(len(all_line))]
    for i in range(len(all_line)):
        VV[i] = all_line[i].split(" ")
        for j in range(3):
            VV[i][j] = int(VV[i][j])
        VV[i][3] = VV[i][3].replace("\n","")
 
    return VV