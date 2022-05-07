import os
import SimpleITK as sitk
from tqdm import tqdm

def resample(in_path, out_path, output_size=[], output_spacing=[]):
    # this will resample the image about the center
    if not os.path.exists(in_path):
        return

    image = sitk.ReadImage(in_path)
    image_center = image.TransformIndexToPhysicalPoint([round(image.GetSize()[i]/2) for i in range(3)])

    new_origin = [image_center[i] - output_size[i]/2* output_spacing[i]*image.GetDirection()[3*i+i] for i in range(3)]

    # resample on image
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetSize(output_size)
    resampler.SetInterpolator(2)
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
    normalizer.SetWindowMaximum(1024)
    normalizer.SetWindowMinimum(-1000)
    image_output = normalizer.Execute(image_output)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_path)
    writer.Execute(image_output)

def main():
    # pwh data directory
    DATA_DIR = "/mnt/DIIR-JK-NAS/data/LungDataPWH/data"
    OUTPUT_DIR = "/mnt/DIIR-JK-NAS/data/LungDataPWH/data_normalized"
    # DATA_DIR = "/home/jacky/data/pwh/original/"
    # OUTPUT_DIR = "/home/jacky/data/pwh/normalized/"
    OUTPUT_SHAPE = [128,128,64]
    OUTPUT_SPACING = [2,2,5]

    files = []

    os.makedirs(OUTPUT_DIR,exist_ok=True)

    pbar1 = tqdm(os.listdir(DATA_DIR))

    for case in pbar1:
        pbar1.set_description(case)
        if not os.path.isdir(os.path.join(DATA_DIR,case)):
            continue

        pbar2 = tqdm(os.listdir(os.path.join(DATA_DIR,case)))
        os.makedirs(os.path.join(OUTPUT_DIR,case),exist_ok=True)

        for file in pbar2:
            pbar2.set_description(file)
            input_path = os.path.join(DATA_DIR,case,file)
            output_path = os.path.join(OUTPUT_DIR,case,file)

            files.append(os.path.join(case,file))
            resample(input_path,output_path,OUTPUT_SHAPE,OUTPUT_SPACING)

    textfile = open(os.path.join(OUTPUT_DIR,"data_list.txt"), "w")
    for element in files:
        textfile.write(element + "\n")
    textfile.close()

if __name__ == "__main__":
    main()