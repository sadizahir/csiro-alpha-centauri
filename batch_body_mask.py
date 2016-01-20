from __future__ import print_function
import os
from subprocess import call

run_mask = "/home/dow170/Dev/milx-view/build/bin/milxMaskVolume"
#run_img = "/data/dow170/Dev/milx-view-pearcey/build/bin/milxImageOperations"
run_img = "/home/dow170/Dev/milx-view/build/bin/milxMaskOperations"
dir_ct = "ct/"
filt = "CT.nii.gz"
mask_suffix = "_BODY"
out_suffix = "_BM"
cou_suffix = "_TABLE"
m_c_suffix = "_BODYONLY"
threshold = -1000

def runCmd(sCmd):
    lcmd=sCmd.split()        
    call(lcmd)

# go through each filename in dir_ct with filt in name
for fn in [f for f in sorted(os.listdir(dir_ct)) if filt in f][0:]:
	fn_components = fn.split(".")
	fn_mask = fn_components[0] + mask_suffix + "." + ".".join(fn_components[1:]) # prepare the mask filename 
	fn_couch = fn_components[0] + cou_suffix + "." + ".".join(fn_components[1:])
	fn_out = fn_components[0] + out_suffix + "." + ".".join(fn_components[1:]) # prepare the output filename
	fn_m_c = fn_components[0] + m_c_suffix + "." + ".".join(fn_components[1:])
	#runCmd(run_mask + " " + dir_ct + fn + " " + dir_ct + fn_mask + " " + dir_ct + fn_out + " " + str(threshold))
	#runCmd(run_mask + " " + dir_ct + fn + " " + dir_ct + fn_m_c + " " + dir_ct + fn_out + " " + str(threshold))


