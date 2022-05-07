import SimpleITK as sitk
import os
from tqdm import tqdm
import pandas as pd
import math
import numpy as np

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
    # try:
    #     df["x_tfm"] = df.apply(lambda row: row.x-tfm.GetDisplacementField().GetPixel(list(map(int,[row.i,row.j,row.k])))[0],axis=1)
    #     df["y_tfm"] = df.apply(lambda row: row.y+tfm.GetDisplacementField().GetPixel(list(map(int,[row.i,row.j,row.k])))[1],axis=1)
    #     df["z_tfm"] = df.apply(lambda row: row.z-tfm.GetDisplacementField().GetPixel(list(map(int,[row.i,row.j,row.k])))[2],axis=1)

    #     df["x_vec"] = df.apply(lambda row: tfm.GetDisplacementField().GetPixel(list(map(int,[row.i,row.j,row.k])))[0],axis=1)
    #     df["y_vec"] = df.apply(lambda row: tfm.GetDisplacementField().GetPixel(list(map(int,[row.i,row.j,row.k])))[1],axis=1)
    #     df["z_vec"] = df.apply(lambda row: tfm.GetDisplacementField().GetPixel(list(map(int,[row.i,row.j,row.k])))[2],axis=1)
    # except:
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

def compute_landmark_mean(df1,df2,tfm=sitk.Transform(3, sitk.sitkIdentity)):
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

def main():
    LANDMARK_DIR = "/mnt/DIIR-JK-NAS/data/lung_data/DIR_LAB_4DCT/unzip"
    
    DATA_DIR_syn = "/mnt/DIIR-JK-NAS/data/lung_data/dir_DFfield"

    ORIGINAL_IMAGE_DIR_vxm = "/mnt/DIIR-JK-NAS/data/lung_data/normalized"
    MOVED_IMAGE_DIR_vxm = "/mnt/DIIR-JK-NAS/data/lung_data/registered/image"
    WRAP_DIR_vxm = "/mnt/DIIR-JK-NAS/data/lung_data/registered/field"
    
    output_df = pd.DataFrame(columns=["case","method","point_set","fixed","moving","tre_moving","sd_moving","tre_moved","sd_moved","rms_moving","rms_moved","l2_moving","corr_moving","MI_moving","l2_moved","corr_moved","MI_moved"])

    WITH_GENERIC_AFFINE = True

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

        # syn 
        print("Computing metrics for SyN registration for case {}".format(case))
        # load warp field
        print("Reading images")
        wrap = sitk.ReadImage(os.path.join(DATA_DIR_syn,"case{}".format(case),"case{}_00-501Warp.nii.gz".format(case)))
        # wrap = sitk.ReadImage(os.path.join(DATA_DIR_syn,"case{}".format(case),"case{}_00-501InverseWarp.nii.gz".format(case)))

        # note that transformation field is in direction of NIFTI coordinate (RAS), ITK coordinate system use RAI direction by default. Here for coding convenience we use ITK system.
        direction = [-1,1,-1]
        # direction = [1,1,1]
        wrap_ = sitk.Compose([sitk.VectorIndexSelectionCast(wrap,i)*direction[i] for i in range(wrap.GetNumberOfComponentsPerPixel())])

        generic_affine = sitk.ReadTransform(os.path.join(DATA_DIR_syn,"case{}".format(case),"case{}_00-500GenericAffine.mat".format(case)))
        generic_affine = generic_affine.GetInverse()

        fixed = sitk.ReadImage(os.path.join(DATA_DIR_syn,"case{}".format(case),"case{}_00.nii".format(case)))
        moving = sitk.ReadImage(os.path.join(DATA_DIR_syn,"case{}".format(case),"case{}_50.nii.gz".format(case)))
        moved = sitk.ReadImage(os.path.join(DATA_DIR_syn,"case{}".format(case),"case{}_00-50Warped.nii.gz".format(case)))

        # transform ijk to xyz
        for pointset, times in landmarks.items():
            for time, landmark in times.items():
                landmarks[pointset][time]["x"] = landmark.apply(lambda row: fixed.TransformIndexToPhysicalPoint(list(map(int,[row.i,row.j,row.k])))[0],axis=1)
                landmarks[pointset][time]["y"] = landmark.apply(lambda row: fixed.TransformIndexToPhysicalPoint(list(map(int,[row.i,row.j,row.k])))[1],axis=1)
                landmarks[pointset][time]["z"] = landmark.apply(lambda row: fixed.TransformIndexToPhysicalPoint(list(map(int,[row.i,row.j,row.k])))[2],axis=1)

        # composite transform
        tfm_disp = sitk.DisplacementFieldTransform(wrap_)
        wrap = tfm_disp.GetDisplacementField() # this line need to be added due to bug will destruct the original wrap image

        tfm = sitk.CompositeTransform(3)
        # The transforms are composed in reverse order with the back being applied first: https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1CompositeTransform.html
        tfm.AddTransform(tfm_disp)
        tfm.AddTransform(generic_affine)

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
        print("Writing warpped image...")
        moved2 = sitk.Resample(moving,tfm)
        sitk.WriteImage(moved2,os.path.join(DATA_DIR_syn,"case{}".format(case),"case{}_00-50Warped_2.nii.gz".format(case)))

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
        
        # # 75 points
        # rms_moving = compute_landmark_rms(landmarks["75"]["T00"],landmarks["75"]["T50"])
        # rms_moved = compute_landmark_rms(landmarks["75"]["T00"],landmarks["75"]["T50"],tfm)
        # mean_moving,sd_moving = compute_landmark_mean(landmarks["75"]["T00"],landmarks["75"]["T50"])
        # mean_moved,sd_moved = compute_landmark_mean(landmarks["75"]["T00"],landmarks["75"]["T50"],tfm)
        # results = {
        #     "case":case,
        #     "method":"syn",
        #     "point_set":"75",
        #     "fixed": "T00",
        #     "moving": "T50",
        #     "tre_moving": mean_moving,
        #     "sd_moving": sd_moving,
        #     "tre_moved": mean_moved,
        #     "sd_moved": sd_moved,
        #     "rms_moving": rms_moving,
        #     "rms_moved": rms_moved
        #     }
        # print("tre_moving (75): {:.2f}".format(mean_moving))
        # print("sd_moving (75): {:.2f}".format(sd_moving))
        # print("tre_moved (75): {:.2f}".format(mean_moved))
        # print("sd moved (75): {:.2f}".format(sd_moved))
        # print("rms_moving (75): {:.2f}".format(rms_moving))
        # print("rms moved (75): {:.2f}".format(rms_moved))
        # output_df = output_df.append(results,ignore_index=True)

        # print("generic affine")
        # print(generic_affine)
        # print(generic_affine.TransformPoint([1,1,1]))

        # 300 points
        print("rms moving")
        rms_moving = compute_landmark_rms(landmarks["300"]["T00"],landmarks["300"]["T50"])
        print("rms moved")
        rms_moved = compute_landmark_rms(landmarks["300"]["T00"],landmarks["300"]["T50"],tfm)
        print("tre moving")
        mean_moving,sd_moving = compute_landmark_mean(landmarks["300"]["T00"],landmarks["300"]["T50"])
        print("tre moved")
        mean_moved,sd_moved = compute_landmark_mean(landmarks["300"]["T00"],landmarks["300"]["T50"],tfm)
        results = {
            "case":case,
            "method":"syn",
            "point_set":"300",
            "fixed": "T00",
            "moving": "T50",
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

        print("compute registartion metric complete for SyN")

        # for vxm registration
        # for k,times in landmarks.items():
        #     for time, landmark in times.items():
        #         print(k, time)
        #         landmark = transform_points(landmark,tfm)

        #         # compute l2 norm
        #         landmark["dist^2"] = landmark.apply(lambda row: (row.x-row.x_tfm)**2+(row.y-row.y_tfm)**2+(row.z-row.z_tfm)**2,axis=1)

        #         results = {
        #             "case":case,
        #             "method":"syn",
        #             "point_set":"image",
        #             "fixed": "T00",
        #             "moving": "T05",
        #             "rms": np.sqrt(landmark["dist^2"].mean())
        #             }

        #         output_df = output_df.append(results,ignore_index=True)

        #         exit()

    print(output_df)

    output_df.to_csv(os.path.join(DATA_DIR_syn,"metrics.csv"),index=False)

if __name__=="__main__":
    main()