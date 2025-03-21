import SimpleITK as sitk
import os
from tqdm import tqdm
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

def mutual_information(img1,img2):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation()
    mi = registration_method.MetricEvaluate(img1,img2)
    return mi*-1

def l2_loss(img1,img2):
    powFilter = sitk.PowImageFilter()
    sq_diff = powFilter.Execute(img1-img2,2)
    statFilter = sitk.StatisticsImageFilter()
    statFilter.Execute(sq_diff)

    l2 = math.sqrt(statFilter.GetMean())

    return l2

def normalize(img):
    normalizer = sitk.IntensityWindowingImageFilter()
    normalizer.SetOutputMaximum(1)
    normalizer.SetOutputMinimum(0)
    normalizer.SetWindowMaximum(2048)
    normalizer.SetWindowMinimum(0)
    image_output = normalizer.Execute(img)

    return image_output

def correlation(img1,img2):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
    corr = registration_method.MetricEvaluate(img1,img2)
    return corr*-1

def transform_points(df,tfm):    
    df["x_tfm"] = df.apply(lambda row: tfm.TransformPoint(list(map(float,[row.x,row.y,row.z])))[0],axis=1)
    df["y_tfm"] = df.apply(lambda row: tfm.TransformPoint(list(map(float,[row.x,row.y,row.z])))[1],axis=1)
    df["z_tfm"] = df.apply(lambda row: tfm.TransformPoint(list(map(float,[row.x,row.y,row.z])))[2],axis=1)

    df["x_vec"] = df.apply(lambda row: row.x_tfm-row.x,axis=1)
    df["y_vec"] = df.apply(lambda row: row.y_tfm-row.y,axis=1)
    df["z_vec"] = df.apply(lambda row: row.z_tfm-row.z,axis=1)

    print(df)
    print(
        "x_vec: {:.2f}".format(df["x_vec"].mean()),
        "y_vec: {:.2f}".format(df["y_vec"].mean()),
        "z_vec: {:.2f}".format(df["z_vec"].mean())
        )

    return df

def compute_landmark_rms(df1,df2,tfm=sitk.Transform(3, sitk.sitkIdentity)):
    df2 = transform_points(df2,tfm) 
    df = pd.concat([df1[["x","y","z"]],df2[["x_tfm","y_tfm","z_tfm"]]],axis=1)

    # compute l2 norm
    df["dist^2"] = df.apply(lambda row: (row.x-row.x_tfm)**2+(row.y-row.y_tfm)**2+(row.z-row.z_tfm)**2,axis=1)

    return np.sqrt(df["dist^2"].mean())

def compute_landmark_tre(df1,df2,tfm=sitk.Transform(3, sitk.sitkIdentity)):
    df2 = transform_points(df2,tfm) 
    df = pd.concat([df1[["x","y","z"]],df2[["x_tfm","y_tfm","z_tfm"]]],axis=1)

    # compute l2 norm
    df["x_diff"] = df.apply(lambda row: row.x-row.x_tfm,axis=1)
    df["y_diff"] = df.apply(lambda row: row.y-row.y_tfm,axis=1)
    df["z_diff"] = df.apply(lambda row: row.z-row.z_tfm,axis=1)

    df["dist"] = df.apply(lambda row: np.sqrt((row.x-row.x_tfm)**2+(row.y-row.y_tfm)**2+(row.z-row.z_tfm)**2),axis=1)

    # print(df)
    print(
        "x_diff: {:.2f}".format(df["x_diff"].mean()),
        "y_diff: {:.2f}".format(df["y_diff"].mean()),
        "z_diff: {:.2f}".format(df["z_diff"].mean())
    )

    return df["dist"].mean(),df["dist"].std()

def plot_2d_hist(fixed,moving,output_path):
    fig,ax = plt.subplots(figsize =(10, 7))
    # Creating plot
    # ax.set_xlabel()
    plt.hist2d(sitk.GetArrayFromImage(fixed).flatten(), sitk.GetArrayFromImage(moving).flatten(),bins=(256, 256),cmap="jet")
    plt.axline(xy1=[1,1], slope=1, linestyle="--", color="red",linewidth=1.5)
    plt.title("Histogram 2D")
    
    # save plot
    plt.savefig(output_path)
    plt.close("all")

def main():
    LANDMARK_DIR = "/mnt/DIIR-JK-NAS/data/lung_data/DIR_LAB_4DCT/unzip"

    DATA_DIR_syn = "/mnt/DIIR-JK-NAS/data/lung_data/dir_DFfield"
    DATA_DIR_vxm = "/mnt/DIIR-JK-NAS/data/lung_data/registered/"

    ORIGINAL_IMAGE_DIR_vxm = "/mnt/DIIR-JK-NAS/data/lung_data/normalized"
    
    output_df = pd.DataFrame(columns=["case","method","point_set","fixed","moving","tre_moving","sd_moving","tre_moved","sd_moved","rms_moving","rms_moved","l2_moving","corr_moving","MI_moving","l2_moved","corr_moved","MI_moved"])

    VXM=False

    for case in range(1,11)[0:1]:
        print("***********************************")
        print("Reading landmarks case {}".format(case))
        landmarks = {"300":{},"75":{}}
        # load 300 landmarks
        times = ["00","50"]    

        for time in times:
            if case < 6:
                path = os.path.join(LANDMARK_DIR,"Case{}".format(case),"ExtremePhases","Case{}_300_T{}_xyz.txt".format(case,time))
            else:
                path = os.path.join(LANDMARK_DIR,"Case{}".format(case),"extremePhases","case{}_dirLab300_T{}_xyz.txt".format(case,time))
            landmarks["300"].update({"T{}".format(time): pd.read_csv(path,names=["i","j","k"],delimiter='\s+')})

        # load 75 landmarks
        times = ["00","10","20","30","40","50"]   
        for time in times:
            path = os.path.join(LANDMARK_DIR,"Case{}".format(case),"Sampled4D","case{}_4D-75_T{}.txt".format(case,time))
            landmarks["75"].update({"T{}".format(time): pd.read_csv(path,names=["i","j","k"],sep='\s+')})

        """
        SYN transform
        """

        # syn 
        print("Computing metrics for SyN registration for case {}".format(case))
        # load warp field
        print("Reading images")
        wrap = sitk.ReadImage(os.path.join(DATA_DIR_syn,"case{}".format(case),"case{}_00-501Warp.nii.gz".format(case)))

        # note that transformation field is in direction of NIFTI coordinate (RAS), ITK coordinate system use RAI direction by default. Here for coding convenience we use ITK system.
        direction = [-1,-1,-1]
        wrap_ = sitk.Compose([sitk.VectorIndexSelectionCast(wrap,i)*direction[i] for i in range(wrap.GetNumberOfComponentsPerPixel())])

        generic_affine = sitk.ReadTransform(os.path.join(DATA_DIR_syn,"case{}".format(case),"case{}_00-500GenericAffine.mat".format(case)))
        # generic_affine = generic_affine.GetInverse()

        fixed = sitk.ReadImage(os.path.join(DATA_DIR_syn,"case{}".format(case),"case{}_00.nii".format(case)))
        moving = sitk.ReadImage(os.path.join(DATA_DIR_syn,"case{}".format(case),"case{}_50.nii.gz".format(case)))
        moved = sitk.ReadImage(os.path.join(DATA_DIR_syn,"case{}".format(case),"case{}_00-50Warped.nii.gz".format(case)))

        # composite transform
        tfm_disp_ = sitk.DisplacementFieldTransform(wrap_)
        tfm_disp_.SetInterpolator(sitk.sitkNearestNeighbor)
        wrap_ = tfm_disp_.GetDisplacementField() # this line need to be added due to bug will destruct the original wrap image

        tfm_ = sitk.CompositeTransform(3)
        # The transforms are composed in reverse order with the back being applied first: https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1CompositeTransform.html
        tfm_.AddTransform(generic_affine.GetInverse()) # for coordinate transform we need the inverse transform
        tfm_.AddTransform(tfm_disp_)

        # tfm_.AddTransform(generic_affine) # for coordinate transform we need the inverse transform


        castFilter = sitk.CastImageFilter()
        castFilter.SetOutputPixelType(sitk.sitkFloat32)
        fixed = castFilter.Execute(fixed)
        moving = castFilter.Execute(moving)
        moved = castFilter.Execute(moved)

        # intensity normalize
        print("Normalizing images...")
        fixed = normalize(fixed)
        moving = normalize(moving)
        moved= normalize(moved)

        # transform image
        tfm_disp = sitk.DisplacementFieldTransform(wrap)
        wrap = tfm_disp.GetDisplacementField() # this line need to be added due to bug will destruct the original wrap image

        tfm = sitk.CompositeTransform(3)
        # The transforms are composed in reverse order with the back being applied first: https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1CompositeTransform.html
        tfm.AddTransform(generic_affine)
        tfm.AddTransform(tfm_disp)

        moved2 = sitk.Resample(moving,tfm)

        # transform ijk to xyz
        for pointset, times in landmarks.items():
            for time, landmark in times.items():
                landmarks[pointset][time]["x"] = landmark.apply(lambda row: fixed.TransformIndexToPhysicalPoint(list(map(int,[row.i,row.j,row.k])))[0],axis=1)
                landmarks[pointset][time]["y"] = landmark.apply(lambda row: fixed.TransformIndexToPhysicalPoint(list(map(int,[row.i,row.j,row.k])))[1],axis=1)
                landmarks[pointset][time]["z"] = landmark.apply(lambda row: fixed.TransformIndexToPhysicalPoint(list(map(int,[row.i,row.j,row.k])))[2],axis=1)

        # plot 2d histogram
        print("Plotting registration histogram...")
        plot_2d_hist(fixed,moving,os.path.join(DATA_DIR_syn,"case{}".format(case),"hist_moving.png"))
        plot_2d_hist(fixed,moved,os.path.join(DATA_DIR_syn,"case{}".format(case),"hist_moved.png"))

        # compute image loss
        print("computing registartion metric for SyN images...")

        results = {
            "case":case,
            "method":"syn",
            "point_set":"image",
            "fixed": "T00",
            "moving": "T50",
            "l2_moving":l2_loss(fixed,moving),
            "corr_moving":correlation(fixed,moving),
            "MI_moving": mutual_information(fixed,moving),
            "l2_moved":l2_loss(fixed,moved),
            "corr_moved":correlation(fixed,moved),
            "MI_moved": mutual_information(fixed,moved)
            }

        output_df = output_df.append(results,ignore_index=True)

        results = {
            "case":case,
            "method":"syn",
            "point_set":"self_wrap",
            "fixed": "T00",
            "moving": "T50",
            "l2_moving":l2_loss(fixed,moving),
            "corr_moving":correlation(fixed,moving),
            "MI_moving": mutual_information(fixed,moving),
            "l2_moved":l2_loss(fixed,moved2),
            "corr_moved":correlation(fixed,moved2),
            "MI_moved": mutual_information(fixed,moved2)
            }

        output_df = output_df.append(results,ignore_index=True)

        """
        transform landmarks
        """
        print("computing registartion metric for SyN landmarks...")
        
        # 75 points
        rms_moving = compute_landmark_rms(landmarks["75"]["T00"],landmarks["75"]["T50"])
        rms_moved = compute_landmark_rms(landmarks["75"]["T00"],landmarks["75"]["T50"],tfm_)
        tre_moving,sd_moving = compute_landmark_tre(landmarks["75"]["T00"],landmarks["75"]["T50"])
        tre_moved,sd_moved = compute_landmark_tre(landmarks["75"]["T00"],landmarks["75"]["T50"],tfm_)
        results = {
            "case":case,
            "method":"syn",
            "point_set":"75",
            "fixed": "T00",
            "moving": "T50",
            "tre_moving": tre_moving,
            "sd_moving": sd_moving,
            "tre_moved": tre_moved,
            "sd_moved": sd_moved,
            "rms_moving": rms_moving,
            "rms_moved": rms_moved
            }
        print("tre_moving (75): {:.2f}".format(tre_moving))
        print("sd_moving (75): {:.2f}".format(sd_moving))
        print("tre_moved (75): {:.2f}".format(tre_moved))
        print("sd moved (75): {:.2f}".format(sd_moved))
        print("rms_moving (75): {:.2f}".format(rms_moving))
        print("rms moved (75): {:.2f}".format(rms_moved))
        output_df = output_df.append(results,ignore_index=True)

        # print("generic affine")
        # print(generic_affine)
        # print(generic_affine.TransformPoint([1,1,1]))

        # 300 points
        print("rms moving")
        rms_moving = compute_landmark_rms(landmarks["300"]["T00"],landmarks["300"]["T50"])
        print("rms moved")
        rms_moved = compute_landmark_rms(landmarks["300"]["T00"],landmarks["300"]["T50"],tfm_)
        print("tre moving")
        tre_moving,sd_moving = compute_landmark_tre(landmarks["300"]["T00"],landmarks["300"]["T50"])
        print("tre moved")
        tre_moved,sd_moved = compute_landmark_tre(landmarks["300"]["T00"],landmarks["300"]["T50"],tfm_)
        results = {
            "case":case,
            "method":"syn",
            "point_set":"300",
            "fixed": "T00",
            "moving": "T50",
            "tre_moving": tre_moving,
            "sd_moving": sd_moving,
            "tre_moved": tre_moved,
            "sd_moved": sd_moved,
            "rms_moving": rms_moving,
            "rms_moved": rms_moved
            }
        print("tre_moving (300): {:.2f}".format(tre_moving))
        print("sd_moving (300): {:.2f}".format(sd_moving))
        print("tre_moved (300): {:.2f}".format(tre_moved))
        print("sd moved (300): {:.2f}".format(sd_moved))
        print("rms_moving (300): {:.2f}".format(rms_moving))
        print("rms moved (300): {:.2f}".format(rms_moved))

        output_df = output_df.append(results,ignore_index=True)
        print("compute registartion metric complete for SyN")

        if VXM:
            """
            VXM transform
            """
            # for vxm registration
            print("Computing metrics for Voxelmorph registration for case {}".format(case))

            os.makedirs(os.path.join(DATA_DIR_vxm,"case{}".format(case),"results"),exist_ok=True)

            for k,times in landmarks.items():
                for time, landmark in times.items():
                    print(k, time)

                    # load warp field
                    print("Reading images")
                    wrap = sitk.ReadImage(os.path.join(DATA_DIR_vxm,"case{}".format(case),"field","case{}_00-case{}_{}.nii".format(case,case,time[-2:])))

                    # convert voxelmorph field to vector image
                    wrap_np = sitk.GetArrayFromImage(wrap).astype(np.float64)
                    wrap_ = sitk.GetImageFromArray(np.moveaxis(wrap_np,0,-1), isVector=True)
                    wrap_.SetOrigin(wrap.GetOrigin())
                    new_dir = [wrap.GetDirection()[int(4*np.floor(i/3)+i%3)] for i in range(9)]
                    wrap_.SetDirection(new_dir)
                    wrap_.SetSpacing(wrap.GetSpacing())
                    wrap = wrap_

                    # note that transformation field is in direction of NIFTI coordinate (RAS), ITK coordinate system use RAI direction by default. Here for coding convenience we use ITK system.
                    direction = [-1,-1,-1]
                    wrap_ = sitk.Compose([sitk.VectorIndexSelectionCast(wrap,i)*direction[i] for i in range(wrap.GetNumberOfComponentsPerPixel())])

                    # generic_affine = sitk.ReadTransform(os.path.join(DATA_DIR_syn,"case{}".format(case),"case{}_00-500GenericAffine.mat".format(case)))

                    fixed = sitk.ReadImage(os.path.join(ORIGINAL_IMAGE_DIR_vxm,"case{}".format(case),"case{}_00.nii".format(case)))
                    moving = sitk.ReadImage(os.path.join(ORIGINAL_IMAGE_DIR_vxm,"case{}".format(case),"case{}_{}.nii".format(case,time[-2:])))
                    moved = sitk.ReadImage(os.path.join(DATA_DIR_vxm,"case{}".format(case),"image","case{}_00-case{}_{}.nii".format(case,case,time[-2:])))

                    # # composite transform
                    tfm_disp_ = sitk.DisplacementFieldTransform(wrap_)
                    tfm_disp_.SetInterpolator(sitk.sitkNearestNeighbor)
                    wrap_ = tfm_disp_.GetDisplacementField() # this line need to be added due to bug will destruct the original wrap image

                    tfm_ = sitk.CompositeTransform(3)
                    # The transforms are composed in reverse order with the back being applied first: https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1CompositeTransform.html
                    tfm_.AddTransform(tfm_disp_)
                    # tfm_.AddTransform(generic_affine.GetInverse()) # for coordinate transform we need the inverse transform

                    castFilter = sitk.CastImageFilter()
                    castFilter.SetOutputPixelType(sitk.sitkFloat32)
                    fixed = castFilter.Execute(fixed)
                    moving = castFilter.Execute(moving)
                    moved = castFilter.Execute(moved)

                    # transform image
                    tfm_disp = sitk.DisplacementFieldTransform(wrap)
                    wrap = tfm_disp.GetDisplacementField() # this line need to be added due to bug will destruct the original wrap image

                    tfm = sitk.CompositeTransform(3)
                    # The transforms are composed in reverse order with the back being applied first: https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1CompositeTransform.html
                    tfm.AddTransform(tfm_disp)
                    # tfm.AddTransform(generic_affine)

                    moved2 = sitk.Resample(moving,tfm)

                    # plot 2d histogram
                    print("Plotting registration histogram...")
                    plot_2d_hist(fixed,moving,os.path.join(DATA_DIR_vxm,"case{}".format(case),"results","hist_moving_case{}_00-case{}_{}.png".format(case,case,time[-2:])))
                    plot_2d_hist(fixed,moved,os.path.join(DATA_DIR_vxm,"case{}".format(case),"results","hist_moved_case{}_00-case{}_{}.png".format(case,case,time[-2:])))

                    # compute image loss
                    print("computing registartion metric for Voxelmorph images...")

                    results = {
                        "case":case,
                        "method":"vxm",
                        "point_set":"image",
                        "fixed": "T00",
                        "moving": time,
                        "l2_moving":l2_loss(fixed,moving),
                        "corr_moving":correlation(fixed,moving),
                        "MI_moving": mutual_information(fixed,moving),
                        "l2_moved":l2_loss(fixed,moved),
                        "corr_moved":correlation(fixed,moved),
                        "MI_moved": mutual_information(fixed,moved)
                        }

                    output_df = output_df.append(results,ignore_index=True)

                    results = {
                        "case":case,
                        "method":"vxm",
                        "point_set":"self_wrap",
                        "fixed": "T00",
                        "moving": time,
                        "l2_moving":l2_loss(fixed,moving),
                        "corr_moving":correlation(fixed,moving),
                        "MI_moving": mutual_information(fixed,moving),
                        "l2_moved":l2_loss(fixed,moved2),
                        "corr_moved":correlation(fixed,moved2),
                        "MI_moved": mutual_information(fixed,moved2)
                        }

                    output_df = output_df.append(results,ignore_index=True)

                    """
                    transform landmarks
                    """
                    print("computing registartion metric for Voxelmorph landmarks...")

                    # 300 points
                    print("rms moving")
                    print(landmarks[k][time])
                    rms_moving = compute_landmark_rms(landmarks[k]["T00"],landmarks[k][time])
                    print("rms moved")
                    rms_moved = compute_landmark_rms(landmarks[k]["T00"],landmarks[k][time],tfm_)
                    print("tre moving")
                    mean_moving,sd_moving = compute_landmark_tre(landmarks[k]["T00"],landmarks[k][time])
                    print("tre moved")
                    mean_moved,sd_moved = compute_landmark_tre(landmarks[k]["T00"],landmarks[k][time],tfm_)
                    results = {
                        "case":case,
                        "method":"vxm",
                        "point_set":k,
                        "fixed": "T00",
                        "moving": time,
                        "tre_moving": mean_moving,
                        "sd_moving": sd_moving,
                        "tre_moved": mean_moved,
                        "sd_moved": sd_moved,
                        "rms_moving": rms_moving,
                        "rms_moved": rms_moved
                        }
                    print("tre_moving (300): {:.2f}".format(mean_moving))
                    print("sd_moving (300): {:.2f}".format(sd_moving))
                    print("tre_moved (300): {:.2f}".format(mean_moved))
                    print("sd moved (300): {:.2f}".format(sd_moved))
                    print("rms_moving (300): {:.2f}".format(rms_moving))
                    print("rms moved (300): {:.2f}".format(rms_moved))

                    output_df = output_df.append(results,ignore_index=True)
                    print("compute registartion metric complete for Voxelmorph")
            break

    print(output_df)

    output_df.to_csv(os.path.join(DATA_DIR_syn,"metrics.csv"),index=False)

if __name__=="__main__":
    main()