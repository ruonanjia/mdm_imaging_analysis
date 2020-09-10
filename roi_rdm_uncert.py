#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Work flow to compute ROI rdms

Created on Mon Dec 23 12:31:48 2019

@author: rj299
"""
import os
import numpy as np

import nipype.interfaces.utility as util  # utility
from nipype import Node, Workflow, MapNode
import nibabel as nib
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.interfaces.io as nio  # Data i/o

#%%
base_root = '/home/rj299/scratch60/mdm_analysis/'
data_root = '/home/rj299/scratch60/mdm_analysis/data_rename'
out_root = '/home/rj299/scratch60/mdm_analysis/output'
#%%
data_dir = data_root
output_dir = os.path.join(out_root, 'imaging')
work_dir = os.path.join(base_root, 'work') # intermediate products

subject_list = [2073, 2550, 2582, 2583, 2584, 2585, 2588, 2592, 2593, 2594, 
           2596, 2597, 2598, 2599, 2600, 2624, 2650, 2651, 2652, 2653, 
           2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 
           2664, 2665, 2666]

con_list = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', 
          '0009', '0010', '0011', '0012']

#%%
infosource = pe.Node(util.IdentityInterface(fields=['subject_id', 'con_id'],),
                  name="infosource")

infosource.iterables = [('subject_id', subject_list)]
infosource.inputs.con_id = con_list

templates = {'contrast': os.path.join(out_root, 'imaging' ,'Sink_resp_uncert_rsa', '1stLevel', '_subject_id_{subject_id}', 'spmT_{con_id}.nii')}

# Flexibly collect data from disk to feed into workflows.
selectfiles = MapNode(nio.SelectFiles(templates,
                      base_directory=out_root),
                      name="selectfiles",
                      iterfield = ['con_id'])
        
#%% Compute ROI RDM function
    
def compute_roi_rdm(in_file,
                    stims,
                    all_masks):
    
    from pathlib import Path
    from nilearn.input_data import NiftiMasker
    import nibabel as nib
    import numpy as np
    
    rdm_out = Path('roi_rdm_uncert.npy').resolve()
    stim_num = len(stims)
    
    # dictionary to store rdms for all rois
    rdm_dict = {}
    
    # loop over all rois
    for mask_name in all_masks.keys():
        mask = all_masks[mask_name]
        masker = NiftiMasker(mask_img=mask)
        
        # initiate matrix
        spmt_allstims_roi= np.zeros((stim_num, np.sum(mask.get_data()).astype(int)))
            
        for (stim_idx, spmt_file) in enumerate(in_file):
            spmt = nib.load(spmt_file)
          
            # get each condition's beta
            spmt_roi = masker.fit_transform(spmt)
            spmt_allstims_roi[stim_idx, :] = spmt_roi
        
        # create rdm
        rdm_roi = 1 - np.corrcoef(spmt_allstims_roi)
        
        rdm_dict[mask_name] = rdm_roi
        
    # save    
    np.save(rdm_out, rdm_dict)
    
    return str(rdm_out)


#%% Compute ROI node
get_roi_rdm = Node(util.Function(
    input_names=['in_file', 'stims', 'all_masks'],
    function=compute_roi_rdm, 
    output_names=['rdm_out']),
    name='get_roi_rdm',
    )    
    
get_roi_rdm.inputs.stims = {'01': 'Med_amb_24', '02': 'Med_amb_50', '03': 'Med_amb_74', 
                            '04': 'Med_risk_25', '05': 'Med_risk_50', '06': 'Med_risk_75', 
                            '07': 'Mon_amb_24', '08': 'Mon_amb_50', '09': 'Mon_amb_74', 
                            '10': 'Mon_risk_25', '11': 'Mon_risk_50', '12': 'Mon_risk_75'
                            }

        
# Masker files
maskfile_vmpfc = os.path.join(output_dir, 'binConjunc_PvNxDECxRECxMONxPRI_vmpfc.nii.gz')
maskfile_vstr = os.path.join(output_dir, 'binConjunc_PvNxDECxRECxMONxPRI_striatum.nii.gz')

maskfile_roi1 = os.path.join(output_dir, 'none_glm_Med_Mon_TFCE_p001_roi1.nii.gz')
maskfile_roi2 = os.path.join(output_dir, 'none_glm_Med_Mon_TFCE_p001_roi2.nii.gz')
maskfile_roi3 = os.path.join(output_dir, 'none_glm_Med_Mon_TFCE_p001_roi3.nii.gz')

maskfile_gilaie_rppc = os.path.join(output_dir, 'Gilaie-DotanEtAl_2014_Study1_rPPC-NScorr-Thres250Voxels_roi.nii')

maskfile_zhang_val_lppc = os.path.join(output_dir, 'zhang_nn_2017_value_sphere_lppc.nii.gz')
maskfile_zhang_val_lofc = os.path.join(output_dir, 'zhang_nn_2017_value_sphere_lofc.nii.gz')
maskfile_zhang_val_rofc = os.path.join(output_dir, 'zhang_nn_2017_value_sphere_rofc.nii.gz')
maskfile_zhang_val_lingual = os.path.join(output_dir, 'zhang_nn_2017_value_sphere_lingual.nii.gz')

maskfile_zhang_sal_acc = os.path.join(output_dir, 'zhang_nn_2017_saliency_sphere_acc.nii.gz')
maskfile_zhang_sal_lprecentral = os.path.join(output_dir, 'zhang_nn_2017_saliency_sphere_lprecentral.nii.gz')
maskfile_zhang_sal_lcaudate = os.path.join(output_dir, 'zhang_nn_2017_saliency_sphere_lcaudate.nii.gz')
maskfile_zhang_sal_rcaudate = os.path.join(output_dir, 'zhang_nn_2017_saliency_sphere_rcaudate.nii.gz')
maskfile_zhang_sal_linsula = os.path.join(output_dir, 'zhang_nn_2017_saliency_sphere_linsula.nii.gz')
maskfile_zhang_sal_rinsula = os.path.join(output_dir, 'zhang_nn_2017_saliency_sphere_rinsula.nii.gz')
maskfile_zhang_sal_lingual = os.path.join(output_dir, 'zhang_nn_2017_saliency_sphere_lingual.nii.gz')

maskfile_levy_amb_striatum = os.path.join(output_dir, 'levy_jn_2010_ambig_sphere_striatum.nii.gz')
maskfile_levy_amb_mpfc = os.path.join(output_dir, 'levy_jn_2010_ambig_sphere_mpfc.nii.gz')
maskfile_levy_amb_pcc = os.path.join(output_dir, 'levy_jn_2010_ambig_sphere_pcc.nii.gz')
maskfile_levy_amb_lamyg = os.path.join(output_dir, 'levy_jn_2010_ambig_sphere_lamyg.nii.gz')
maskfile_levy_amb_sts = os.path.join(output_dir, 'levy_jn_2010_ambig_sphere_sts.nii.gz')

maskfile_levy_risk_striatum = os.path.join(output_dir, 'levy_jn_2010_risk_sphere_striatum.nii.gz')
maskfile_levy_risk_mpfc = os.path.join(output_dir, 'levy_jn_2010_risk_sphere_mpfc.nii.gz')

maskfiles = {'vmpfc': maskfile_vmpfc, 
             'vstr': maskfile_vstr, 
             'med_mon_1': maskfile_roi1, 
             'med_mon_2': maskfile_roi2, 
             'med_mon_3': maskfile_roi3,
             'gilaie_rppc': maskfile_gilaie_rppc,
             'zhang_val_lppc': maskfile_zhang_val_lppc,
             'zhang_val_lofc': maskfile_zhang_val_lofc,
             'zhang_val_rofc': maskfile_zhang_val_rofc,
             'zhang_val_lingual': maskfile_zhang_val_lingual,
             'zhang_sal_acc': maskfile_zhang_sal_acc,
             'zhang_sal_lprecentral': maskfile_zhang_sal_lprecentral,
             'zhang_sal_lcaudate': maskfile_zhang_sal_lcaudate,
             'zhang_sal_rcaudate': maskfile_zhang_sal_rcaudate,
             'zhang_sal_linsula': maskfile_zhang_sal_linsula,
             'zhang_sal_rinsula': maskfile_zhang_sal_rinsula,
             'zhang_sal_lingual': maskfile_zhang_sal_lingual,
             'levy_amb_striatum': maskfile_levy_amb_striatum,
             'levy_amb_mpfc': maskfile_levy_amb_mpfc,
             'levy_amb_pcc': maskfile_levy_amb_pcc,
             'levy_amb_lamyg': maskfile_levy_amb_lamyg,
             'levy_amb_sts': maskfile_levy_amb_sts,
             'levy_risk_striatum': maskfile_levy_risk_striatum,
             'levy_risk_mpfc': maskfile_levy_risk_mpfc
             }

# roi inputs are loaded images
get_roi_rdm.inputs.all_masks = {key_name: nib.load(maskfiles[key_name]) for key_name in maskfiles.keys()}

#%%
wf_roirdm = Workflow(name="roi_rdm_uncert", base_dir=work_dir)

wf_roirdm.connect([
        (infosource, selectfiles, [('subject_id', 'subject_id'), ('con_id', 'con_id')]),
        (selectfiles, get_roi_rdm, [('contrast', 'in_file')]),
        ])

#%%
# Datasink
datasink_rdm = Node(nio.DataSink(base_directory=os.path.join(output_dir, 'Sink_resp_uncert_rsa')),
                                         name="datasink_rdm")
                       

wf_roirdm.connect([
        (get_roi_rdm, datasink_rdm, [('rdm_out', 'rdm_new.@rdm')]),
        ])
#%% 
wf_roirdm.run('Linear', plugin_args = {'n_procs': 1})    
