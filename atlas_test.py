from __future__ import print_function
import os
from subprocess import call
import datetime

run_rigid = "/data/dow170/Dev/milx-view-pearcey/build/bin/milxAliBaba"
run_resample = "/data/dow170/Dev/milx-view-pearcey/build/bin/itkResampleImageITKTransformFilter"
run_mask = "/data/dow170/Dev/milx-view-pearcey/build/bin/milxMaskVolume"
run_nrr = "/data/dow170/Dev/milx-view-pearcey/build/bin/milxImageRegistration_SingleLabelledAtlasToImage"
run_img = "/home/dow170/Dev/milx-view/build/bin/milxImageOperations"

def runCmd(sCmd):
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        lcmd=sCmd.split()
        print(lcmd)
        call(lcmd)

#fixed = "ct/B006_PLAN_CT.nii.gz"
#mov = "ct/B037_PLAN_CT.nii.gz"
#bone = "ct/B037_PLAN_CT_CTV.nii.gz"
#tfm = "atlas_bb/B006_B037.tfm"
#resRig = "atlas_bb/B006_B037.nii.gz"
#resRigBone = "atlas_bb/B006_B037_nrr.nii.gz"
#sub = "atlas_bb/B006_B037"

dir_ct = "ct/"
dir_tfm = "atlas_bb/"
dir_nrr = "atlas_lb/"
filt = "_BM"
filt_nrr = "labelling0"
moving_lb_suffix = "_CTV"
bo_lb_suffix = "_BODYONLY"
res_tfm_suffix = "_TFM"
res_rig_suffix = "_RIG"
res_rig_lb_suffix = "_RIG_CTV"
res_rig_bo_suffix = "_RIG_BODYONLY"
res_rig_bm_suffix = "_RIG_BM"
res_dil_suffix = "_ATL_CTV_DILATE"
threshold = -1000

#runCmd(RunRigid + " -a 0 -c 3 -n 0 -f " + fixed + " -m " + mov + " -l " + tfm + " -t rigid --save-moving " + resRig)
#runCmd(RunResample + " " + bone + " " + tfm + " 1 " + resRigBone + " " + fixed)
#runCmd(RunReg + " " + fixed + " " + str(0) + " " + resRig + " " + resRigBone + " " + sub + " " + "13")

# list of base CTs to process
base_cts = [f for f in sorted(os.listdir(dir_ct)) if filt in f][0:]

# reference to an atlas image which will be fitted to other images
moving = dir_ct + base_cts[0] 
moving_lb = dir_ct + base_cts[0].split(".")[0][:-len(filt)] + moving_lb_suffix + "." + ".".join(base_cts[0].split(".")[1:])

rigid = False
rigid_bone = False
nrr = False
dilate = True

for fn in base_cts[1:]: # leave the atlas out and use it on the others
    fn_components = fn.split('.')
    case = fn_components[0].split('_')[0]
    
    if rigid: # Apply Rigid         
        # the target image that the atlas will transform to
        fixed = dir_ct + fn
        # where the rigid transform will be saved    
        res_tfm = dir_tfm + fn_components[0] + res_tfm_suffix + "." + "tfm"
        # where the rigid result will be saved
        res_rig = dir_tfm + fn_components[0] + res_rig_suffix + "." + ".".join(fn_components[1:]) 
        # run the command
        runCmd(run_rigid + " -a 0 -c 3 -n 0 -f " + fixed + " -m " + moving + " -l " + res_tfm + " -t rigid --save-moving " + res_rig)
    
    if rigid_bone: # Apply Rigid to moving label as well  
        fixed = dir_ct + fn
        fixed_bo = dir_ct + fn_components[0][:-len(filt)] + bo_lb_suffix + "." + ".".join(base_cts[0].split(".")[1:])
        tfm = dir_tfm + fn_components[0] + res_tfm_suffix + "." + "tfm"
        res_rig = dir_tfm + fn_components[0] + res_rig_suffix + "." + ".".join(fn_components[1:]) 
        res_rig_lb = dir_tfm + fn_components[0] + res_rig_lb_suffix + "." + ".".join(fn_components[1:])
        res_rig_bo = dir_tfm + fn_components[0] + res_rig_bo_suffix + "." + ".".join(fn_components[1:])
        res_rig_bm = dir_tfm + fn_components[0] + res_rig_bm_suffix + "." + ".".join(fn_components[1:])
        runCmd(run_resample + " " + moving_lb + " " + tfm + " 1 " + res_rig_lb + " " + fixed)
        runCmd(run_resample + " " + fixed_bo + " " + tfm + " 1 " + res_rig_bo + " " + fixed)
        runCmd(run_mask + " " + res_rig + " " + res_rig_bo + " " + res_rig_bm + " " + str(threshold))
    
    
    if nrr: # Apply Non-Rigid Registration
        fixed = dir_ct + fn
        res_rig_bm = dir_tfm + fn_components[0] + res_rig_bm_suffix + "." + ".".join(fn_components[1:])
        res_rig_lb = dir_tfm + fn_components[0] + res_rig_lb_suffix + "." + ".".join(fn_components[1:])        
        runCmd(run_nrr + " " + fixed + " "  + str(0) + " " + res_rig_bm + " " + res_rig_lb + " " + case + "_" + " " + str(13))

if dilate:
    # List of NRRs to dilate
    base_nrrs = [f for f in sorted(os.listdir(dir_nrr)) if filt_nrr in f][0:]
    
    for fn in base_nrrs:
        fn_components = fn.split('.')
        case = fn_components[0].split('_')[0]
        in_dil = dir_nrr + fn
        res_dil = dir_nrr + case + res_dil_suffix + "." + ".".join(fn_components[1:])
        print(run_img + " " + "-g" + " " + in_dil + " " + str(3) + " " + res_dil)
        runCmd(run_img + " " + "-g" + " " + in_dil + " " + str(3) + " " + res_dil)
        runCmd(run_img + " " + "-g" + " " + res_dil + " " + str(3) + " " + res_dil)
        runCmd(run_img + " " + "-g" + " " + res_dil + " " + str(3) + " " + res_dil)
        