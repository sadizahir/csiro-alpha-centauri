from __future__ import print_function
import os
from subprocess import call
import datetime
from joblib import Parallel, delayed
import sys


run_rigid = "/data/dow170/Dev/milx-view-pearcey/build/bin/milxAliBaba"
run_resample = "/data/dow170/Dev/milx-view-pearcey/build/bin/itkResampleImageITKTransformFilter"
run_mask = "/data/dow170/Dev/milx-view-pearcey/build/bin/milxMaskVolume"
run_nrr = "/data/dow170/Dev/milx-view-pearcey/build/bin/milxImageRegistration_SingleLabelledAtlasToImage"
run_img = "/home/dow170/Dev/milx-view/build/bin/milxImageOperations"
run_vote = "/data/dow170/Dev/milx-view-pearcey/build/bin/milxMaskVoting"
run_sims = "/data/dow170/Dev/milx-view-pearcey/build/bin/milxComputeSimilarity"

def runCmd(sCmd):
        #print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        lcmd=sCmd.split()
        #print(lcmd)
        call(lcmd)

dir_ct = "ct/"
dir_tfm = "atlas_bb/"
dir_nrr = "atlas/"
dir_voted = "atlas_lb/"
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
mask_voting_suffix = "_ATL_CTV_VOTE"
mask_staple_suffix = "_ATL_CTV_STPL"
threshold = -1000
jobs = 15
rigid = False
rigid_bone = False
nrr = False
nrr_p = False
rename = False
sims = True
dilate = False

if __name__ == "__main__":
    # This is going to be the list of "base" CT images that are going to be processed.
    base_cts = [f for f in sorted(os.listdir(dir_ct)) if filt in f][0:]
        
    # This is going to be the "fixed" image that all the other CTs are going to 
    # try to fit to. Essentially, you will generate a contour for this image via
    # ATLAS registration.
    fixed_i = int(sys.argv[1])
    fixed = dir_ct + base_cts[fixed_i] 
    fixed_case = base_cts[fixed_i].split('.')[0].split('_')[0]
    
    payloads = []
    
    # Iterate through all the Base CTs
    for i, fn in enumerate(base_cts): 
        
        if i == fixed_i: # if we're on the "target" CT
            continue # move on to the next CT    
        
        # Seperate the filename from the extension and grab the case ID
        fn_components = fn.split('.') 
        case = fn_components[0].split('_')[0]
        
        if rigid: # Apply Rigid transform from a moving CT to the fixed CT
            moving = dir_ct + fn_components[0] + ".nii.gz"
          
            # where the rigid transform will be saved    
            res_tfm = dir_tfm + case + "_TO_" + fixed_case + "_" + \
            '_'.join(fn_components[0].split('_')[1:]) + res_tfm_suffix + \
            "." + "tfm"
            
            # where the rigid result will be saved
            res_rig = dir_tfm + case + "_TO_" + fixed_case + "_" + \
            '_'.join(fn_components[0].split('_')[1:]) + res_rig_suffix + \
            "." + ".".join(fn_components[1:]) 
            
            # Apply Rigid Registration from moving to fixed, outputting the
            # resultant "moved" image to "res_rig" and outputting the transform
            # used to "res_tfm"
            runCmd(run_rigid + " -a 0 -c 3 -n 0 -f " + fixed + " -m " + \
            moving + " -l " + res_tfm + " -t rigid --save-moving " + res_rig)
        
        if rigid_bone: # Apply Rigid to moving label as well  
            # The moving image
            moving = dir_ct + fn_components[0] + ".nii.gz"
            
            # The label of the moving image
            moving_lb = dir_ct + '_'.join(fn_components[0].split('_')[:-1]) + \
            moving_lb_suffix + ".nii.gz"
            
            # The "bodyonly" mask for the moving image; we need to transform this
            # one also (using the earlier generated .tfm) so that we can re-mask
            # the rigid registration output from the previous step. The reason for
            # this is to cut out any artifacts created by milx, which is expecting
            # an MR, but will get a CT.
            moving_bo = dir_ct + fn_components[0][:-len(filt)] + bo_lb_suffix + \
            "." + ".".join(fn.split(".")[1:])
            
            # the transform that was created in the previous step
            tfm = dir_tfm + case + "_TO_" + fixed_case + "_" + \
            '_'.join(fn_components[0].split('_')[1:]) + res_tfm_suffix + \
            "." + "tfm"
            
            # The Rigid-Registered CT output file from the previous step!
            res_rig = dir_tfm + case + "_TO_" + fixed_case + "_" + \
            '_'.join(fn_components[0].split('_')[1:]) + res_rig_suffix + \
            "." + ".".join(fn_components[1:]) 
            
            # This will be the filename of the moving label after the rigid
            # registration transform from the previous step has been done to it
            res_rig_lb = dir_tfm + case + "_TO_" + fixed_case + "_" + \
            '_'.join(fn_components[0].split('_')[1:]) + res_rig_lb_suffix + \
            "." + ".".join(fn_components[1:]) 
            
            # This will be the filename of the bodyonly mask (label) after the
            # rigid registration transform from the previous step has been applied
            res_rig_bo = dir_tfm + case + "_TO_" + fixed_case + "_" + \
            '_'.join(fn_components[0].split('_')[1:]) + res_rig_bo_suffix + \
            "." + ".".join(fn_components[1:]) 
            
            # This will be the filename of the image that is the result of the
            # registered image (generated from the previous step) masked by the
            # transformed bodyonly mask.        
            res_rig_bm = dir_tfm + case + "_TO_" + fixed_case + "_" + \
            '_'.join(fn_components[0].split('_')[1:]) + res_rig_bm_suffix + \
            "." + ".".join(fn_components[1:]) 
            
            # Apply the Rigid transform generated in the previous step (that can
            # transform the "moving" to the "fixed") to the moving label as well.
            runCmd(run_resample + " " + moving_lb + " " + tfm + " 1 " + \
            res_rig_lb + " " + fixed)
            
            # Also apply the Rigid transform to the Body Only Mask!
            runCmd(run_resample + " " + moving_bo + " " + tfm + " 1 " + \
            res_rig_bo + " " + fixed)
    
            # Finally, mask the Registered stuff from Outer Step 1
            runCmd(run_mask + " " + res_rig + " " + res_rig_bo + " " + \
            res_rig_bm + " " + str(threshold))
        
       
        if nrr: # Apply Non-Rigid Registration
            # this is body-masked rigid that was created last time!        
            res_rig_bm = dir_tfm + case + "_TO_" + fixed_case + "_" + \
            '_'.join(fn_components[0].split('_')[1:]) + res_rig_bm_suffix + \
            "." + ".".join(fn_components[1:]) 
            
            # this was the rigid label that was made last time
            res_rig_lb = dir_tfm + case + "_TO_" + fixed_case + "_" + \
            '_'.join(fn_components[0].split('_')[1:]) + res_rig_lb_suffix + \
            "." + ".".join(fn_components[1:]) 
    
            payload = run_nrr + " " + fixed + " "  + str(0) + " " + res_rig_bm + " " + \
            res_rig_lb + " " + dir_nrr + case + "_TO_" + fixed_case + "_" + " " + str(13)
            
            if not nrr_p:
                runCmd(payload)
        
        if nrr_p:
            payloads.append(payload)
    
    if nrr_p:
        Parallel(n_jobs=jobs)(delayed(runCmd)(p) for p in payloads)            
        
    
    if rename: # renames all the stuff
        filt_nrr_fixed = fixed_case + "_" + filt_nrr
        base_atlases = [dir_nrr + f for f in sorted(os.listdir(dir_nrr)) if \
            filt_nrr_fixed in f][0:]
        base_atlases_conc = ' '.join(base_atlases)
        voting_out = dir_voted + fixed_case + mask_voting_suffix + ".nii.gz"
        staple_out = dir_voted + fixed_case + mask_staple_suffix + ".nii.gz"
        command1 = run_vote + " -v " + voting_out + " -i " + base_atlases_conc
        command2 = run_vote + " -s " + staple_out + " -i " + base_atlases_conc
        runCmd(command2)
        runCmd(command1)
        
    if sims: # compute all the similarities between the stapled and the reals
         for i, fn in enumerate(base_cts):
             # Seperate the filename from the extension and grab the case ID
             fn_components = fn.split('.') 
             case = fn_components[0].split('_')[0]
             print("Case: " + case)
             staple_img = dir_voted + case + mask_voting_suffix + ".nii.gz"
             ground_img = dir_ct + case + "_PLAN_CT" + moving_lb_suffix + ".nii.gz"
             command = run_sims + " -i " + staple_img + " -r " + ground_img
             runCmd(command)
             
    
    if dilate:
        # List of NRRs to dilate
        base_nrrs = [f for f in sorted(os.listdir(dir_nrr)) if filt_nrr in f][0:]
        
        for fn in base_nrrs:
            print(fn)
            #fn_components = fn.split('.')
            #case = fn_components[0].split('_')[0]
            #in_dil = dir_nrr + fn
            #res_dil = dir_nrr + case + res_dil_suffix + "." + ".".join(fn_components[1:])
            #print(run_img + " " + "-g" + " " + in_dil + " " + str(3) + " " + res_dil)
            #runCmd(run_img + " " + "-g" + " " + in_dil + " " + str(3) + " " + res_dil)
            #runCmd(run_img + " " + "-g" + " " + res_dil + " " + str(3) + " " + res_dil)
            #runCmd(run_img + " " + "-g" + " " + res_dil + " " + str(3) + " " + res_dil)
            