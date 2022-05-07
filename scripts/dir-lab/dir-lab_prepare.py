import os
import SimpleITK as sitk
from tqdm import tqdm

def resample(in_path, out_path, output_size=[], output_spacing=[]):
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
    normalizer.SetWindowMaximum(2048)
    normalizer.SetWindowMinimum(0)
    image_output = normalizer.Execute(image_output)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_path)
    writer.Execute(image_output)

def main():
    # dir-lab data directory
    # DATA_DIR = "/home/jacky/DIIR-JK-NAS/data/lung_data/original"
    # OUTPUT_DIR = "/home/jacky/DIIR-JK-NAS/data/lung_data/normalized"
    DATA_DIR = "/home/jacky/data/dir-lab/original/"
    OUTPUT_DIR = "/home/jacky/data/dir-lab/normalized/"
    OUTPUT_SHAPE = [128,128,64]
    OUTPUT_SPACING = [2,2,5]

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

            files.append(os.path.join(case,file))
            # resample(input_path,output_path,OUTPUT_SHAPE,OUTPUT_SPACING)

    textfile = open(os.path.join(OUTPUT_DIR,"data_list.txt"), "w")
    for element in files:
        textfile.write(element + "\n")
    textfile.close()

if __name__ == "__main__":
    main()