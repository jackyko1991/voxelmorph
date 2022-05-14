import os
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import pandas as pd

def resample(in_path, out_path, output_size=[], output_spacing=[],interpolator="linear"):
    # this will resample the image about the center
    if not os.path.exists(in_path):
        return

    image = sitk.ReadImage(in_path)

    # tqdm.write("input image path:".format(in_path))
    # tqdm.write("input image size: ({:.2f},{:.2f},{:.2f})".format(image.GetSize()[0],image.GetSize()[1],image.GetSize()[2]))
    # tqdm.write("input image spacing: ({:.2f},{:.2f},{:.2f})".format(image.GetSpacing()[0],image.GetSpacing()[1],image.GetSpacing()[2]))
    # tqdm.write("input image origin: ({:.2f},{:.2f},{:.2f})".format(image.GetOrigin()[0],image.GetOrigin()[1],image.GetOrigin()[2]))

    image_center = image.TransformIndexToPhysicalPoint([round(image.GetSize()[i]/2) for i in range(3)])

    # tqdm.write("input image center: ({:.2f},{:.2f},{:.2f})".format(image_center[0],image_center[1],image_center[2]))

    new_origin = [image_center[i] - output_size[i]/2* output_spacing[i]*image.GetDirection()[3*i+i] for i in range(3)]

    # tqdm.write("output image origin: ({:.2f},{:.2f},{:.2f})".format(new_origin[0],new_origin[1],new_origin[2]))

    # resample on image
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetSize(output_size)
    if interpolator == "linear":
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolator == "NN":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(new_origin)
    resampler.SetOutputDirection(image.GetDirection())
    image_output =  resampler.Execute(image)

    # cast the image to float
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkFloat32)
    image_output = caster.Execute(image_output)

    # normalize the image
    normalizer = sitk.IntensityWindowingImageFilter()
    normalizer.SetOutputMaximum(1)
    normalizer.SetOutputMinimum(0)
    normalizer.SetWindowMaximum(2048)
    normalizer.SetWindowMinimum(0)
    image_output = normalizer.Execute(image_output)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_path)
    writer.Execute(image_output)

def landmark_mask(img_file, lm_file, mask_file):
    tqdm.write("Creating mask file for {}...".format(img_file))

    imgf = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(imgf).astype(np.float32)
    mask = np.zeros(img.shape, dtype=np.int)
    file = lm_file
    with open(file, 'r') as f:
        data = f.readlines()
        for line in data:
            #print(line)
            numbers = line.split()
            numbers = np.array(numbers).astype(np.float).astype(np.int)
            mask[numbers[2],numbers[1],numbers[0]] = 1
            # coordinate 2,1,0 is correct
            #print(numbers)

    # landmark = pd.read_csv(lm_file,names=["i","j","k"],delimiter='\s+')
    img_mask = sitk.GetImageFromArray(mask)
    img_mask.SetSpacing(imgf.GetSpacing())
    img_mask.SetDirection(imgf.GetDirection())
    img_mask.SetOrigin(imgf.GetOrigin())

    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkInt8)
    img_mask = castFilter.Execute(img_mask)

    sitk.WriteImage(img_mask, mask_file)
    tqdm.write("Mask file saved at {}".format(mask_file))

def main():
    # dir-lab data directory
    DATA_DIR = "/home/jacky/DIIR-JK-NAS/data/lung_data/original"
    LANDMARK_DIR = "/home/jacky/DIIR-JK-NAS/data/lung_data/DIR_LAB_4DCT/unzip"
    OUTPUT_DIR = "/home/jacky/DIIR-JK-NAS/data/lung_data/normalized"
    # DATA_DIR = "/home/jacky/data/dir-lab/original/"
    # OUTPUT_DIR = "/home/jacky/data/dir-lab/normalized/"
    OUTPUT_SHAPE = [128,128,128]
    OUTPUT_SPACING = [2.5,2.5,2.5]

    files = []

    pbar1 = tqdm(os.listdir(DATA_DIR))
    for case in pbar1:
        pbar1.set_description(case)
        pbar2 = tqdm(os.listdir(os.path.join(DATA_DIR,case)))
        os.makedirs(os.path.join(OUTPUT_DIR,case),exist_ok=True)

        for file in pbar2:
            pbar2.set_description(file)
            input_path = os.path.join(DATA_DIR,case,file)
            output_path = os.path.join(OUTPUT_DIR,case,file)

            files.append(os.path.join(OUTPUT_DIR,case,file))
            # files.append(os.path.join(case,file))
            # resample(input_path,output_path,OUTPUT_SHAPE,OUTPUT_SPACING)

        # for time in ["00","50"]:
        #     file = "{}_{}.nii".format(case,time)
        #     input_path = os.path.join(DATA_DIR,case,file)
        #     output_path = os.path.join(DATA_DIR,case,"{}_mask.nii".format(file.split(".")[0]))

        #     # create mask output from 300 point dataset
        #     landmark_path = os.path.join(LANDMARK_DIR,"Case{}".format(case[4:]),"ExtremePhases","Case{}_300_T{}_xyz.txt".format(case[4],time))

        #     landmark_mask(input_path,landmark_path,output_path)

        #     # output resample result
        #     input_path = os.path.join(DATA_DIR,case,"{}_mask.nii".format(file.split(".")[0]))
        #     output_path = os.path.join(OUTPUT_DIR,case,"{}_mask.nii".format(file.split(".")[0]))
        #     resample(input_path,output_path,OUTPUT_SHAPE,OUTPUT_SPACING,interpolator="NN")
        # break

    # textfile = open(os.path.join(OUTPUT_DIR,"data_list.txt"), "w")
    # for element in files:
    #     textfile.write(element + "\n")
    # textfile.close()

if __name__ == "__main__":
    main()