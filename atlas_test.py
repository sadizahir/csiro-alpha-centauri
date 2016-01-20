import os
from subprocess import call
import datetime

RunRigid = "/data/dow170/Dev/milx-view-pearcey/build/bin/milxAliBaba"
RunResample = "/data/dow170/Dev/milx-view-pearcey/build/bin/itkResampleImageITKTransformFilter"
RunReg = "/data/dow170/Dev/milx-view-pearcey/build/bin/milxImageRegistration_SingleLabelledAtlasToImage"

def runCmd(sCmd):
        print datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lcmd=sCmd.split()
        print lcmd
        call(lcmd)

fixed = "ct/B006_PLAN_CT.nii.gz"
mov = "ct/B037_PLAN_CT.nii.gz"
bone = "ct/B037_PLAN_CT_CTV.nii.gz"
tfm = "atlas_bb/B006_B037.tfm"
resRig = "atlas_bb/B006_B037.nii.gz"
resRigBone = "atlas_bb/B006_B037_nrr.nii.gz"
sub = "atlas_bb/B006_B037"

runCmd(RunRigid + " -a 0 -c 3 -n 0 -f " + fixed + " -m " + mov + " -l " + tfm + " -t rigid --save-moving " + resRig)

runCmd(RunResample + " " + bone + " " + tfm + " 1 " + resRigBone + " " + fixed)

runCmd(RunReg + " " + fixed + " " + str(0) + " " + resRig + " " + resRigBone + " " + sub + " " + "13")


