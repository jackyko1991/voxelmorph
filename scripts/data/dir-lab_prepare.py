import os
import SimpleITK as sitk
from tqdm import tqdm

def resample(in_path, out_path, output_size=[], output_spacing=[]):
    # this will resample the image about the center
    if not os.path.exists(in_path):
        return

    image = sitk.ReadImage(in_path)

    # tqdm.write("input image size: ({:.2f},{:.2f},{:.2f})".format(image.GetSize()[0],image.GetSize()[1],image.GetSize()[2]))
    # tqdm.write("input image spacing: ({:.2f},{:.2f},{:.2f})".format(image.GetSpacing()[0],image.GetSpacing()[1],image.GetSpacing()[2]))

    image_center = image.TransformIndexToPhysicalPoint([round(image.GetSize()[i]/2) for i in range(3)])

    new_origin = [image_center[i] - output_size[i]/2* image.GetSpacing()[i]*image.GetDirection()[3*i+i] for i in range(3)]

    # resample on image
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetSize(output_size)
    resampler.SetInterpolator(2)
    resampler.SetOutputOrigin(new_origin)
    resampler.SetOutputDirection(image.GetDirection())
    image_output =  resampler.Execute(image)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_path)
    writer.Execute(image_output)

def main():
    # dir-lab data directory
    DATA_DIR = "/home/jacky/DIIR-JK-NAS/data/lung_data/original"
    OUTPUT_DIR = "/home/jacky/DIIR-JK-NAS/data/lung_data/normalized"
    OUTPUT_SHAPE = [256,256,128]
    OUTPUT_SPACING = [1,1,2.5]

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

            files.append(output_path)
            resample(input_path,output_path,OUTPUT_SHAPE,OUTPUT_SPACING)

    textfile = open(os.path.join(OUTPUT_DIR,"data_list.txt"), "w")
    for element in files:
        textfile.write(element + "\n")
    textfile.close()

if __name__ == "__main__":
    main()