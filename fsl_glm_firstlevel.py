# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import os
# import glob
import pandas as pd

#%%
base_root = '/home/rj299/project/mdm_analysis/'
data_root = '/home/rj299/project/mdm_analysis/data_rename'
out_root = '/home/rj299/project/mdm_analysis/output'

#%%
from nipype.pipeline import engine as pe
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces import fsl, utility as niu, io as nio
from nipype.workflows.fmri.fsl.preprocess import create_susan_smooth
from nipype.interfaces.io import BIDSDataGrabber
from niworkflows.interfaces.bids import DerivativesDataSink# as BIDSDerivativesy


#%%
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
# !export OPENBLAS_NUM_THREADS=1
data_dir = data_root
output_dir = os.path.join(out_root, 'imaging')
work_dir = os.path.join(base_root, 'work') # intermediate products

# subject_list = [2583, 2588]
# task_list = [1,2,3,4,5,6,7,8]

subject_list = [2073]
task_list = [6]

fwhm = 6
tr = 1

# Map field names to individual subject runs.
infosource = pe.Node(niu.IdentityInterface(fields=['subject_id', 'task_id'],),
                  name="infosource")

infosource.iterables = [('subject_id', subject_list), 
                        ('task_id', task_list)]

#%%
def _bids2nipypeinfo(in_file, events_file, regressors_file,
                     regressors_names=None,
                     motion_columns=None,
                     decimals=3, amplitude=1.0):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base.support import Bunch

    # Process the events file
    events = pd.read_csv(events_file, sep=r'\s+')

    bunch_fields = ['onsets', 'durations', 'amplitudes']

    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]

    out_motion = Path('motion.par').resolve()
    
    regress_data = pd.read_csv(regressors_file, sep=r'\s+')
    np.savetxt(out_motion, regress_data[motion_columns].values, '%g')
#     np.savetxt(out_motion, regress_data[motion_columns].fillna(0.0).values, '%g')
    
    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))

    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']

    runinfo = Bunch(
        scans=in_file,
        conditions=list(set(events.trial_type.values)),
        **{k: [] for k in bunch_fields})

    for condition in runinfo.conditions:
        event = events[events.trial_type.str.match(condition)]

        runinfo.onsets.append(np.round(event.onset.values, 3).tolist())
        runinfo.durations.append(np.round(event.duration.values, 3).tolist())
        if 'amplitudes' in events.columns:
            runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
        else:
            runinfo.amplitudes.append([amplitude] * len(event))

    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        runinfo.regressors = regress_data[regressors_names].fillna(0.0).values.T.tolist()

    return [runinfo], str(out_motion)

#%%
templates = {'func': os.path.join(data_root, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-{task_id}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
             'mask': os.path.join(data_root, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-{task_id}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'),
             'regressors': os.path.join(data_root, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-{task_id}_desc-confounds_regressors.tsv'),
             'events': os.path.join(out_root, 'event_files', 'sub-{subject_id}_task-{task_id}_cond.csv')}

# Flexibly collect data from disk to feed into workflows.
selectfiles = pe.Node(nio.SelectFiles(templates,
                               base_directory=data_root),
                   name="selectfiles")
        
# Extract motion parameters from regressors file
runinfo = pe.Node(niu.Function(
    input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names', 'motion_columns'],
    function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
    name='runinfo')

# Set the column names to be used from the confounds file
# reference a paper from podlrack lab
runinfo.inputs.regressors_names = ['std_dvars', 'framewise_displacement'] + \
                                  ['a_comp_cor_%02d' % i for i in range(6)]
                                  

runinfo.inputs.motion_columns   = ['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2'] +\
                                  ['trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2'] +\
                                  ['trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2'] +\
                                  ['rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2'] +\
                                  ['rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2'] +\
                                  ['rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']

# SUSAN smoothing
susan = create_susan_smooth()
susan.inputs.inputnode.fwhm = fwhm

# create workflow
workflow = pe.Workflow(name='firstLevel_RA_MDM',base_dir=work_dir)

# set contrasts
cont1 = ['amb', 'T', ['amb', 'risk'], [1, 0]]
cont2 = ['risk', 'T', ['amb', 'risk'], [0, 1]]
# need to add if logic for mon and med condition
contrasts = [cont1, cont2]

# 
l1_spec = pe.Node(SpecifyModel(
    parameter_source='FSL',
    input_units='secs',
    high_pass_filter_cutoff=120,
    time_repetition = tr,
), name='l1_spec')

# l1_model creates a first-level model design
l1_model = pe.Node(fsl.Level1Design(
    bases={'dgamma': {'derivs': True}}, # adding temporal derivative of double gamma
    model_serial_correlations=True,
    interscan_interval = tr,
    contrasts=contrasts
    # orthogonalization=orthogonality,
), name='l1_model')

# feat_spec generates an fsf model specification file
feat_spec = pe.Node(fsl.FEATModel(), name='feat_spec')

# feat_fit actually runs FEAT
feat_fit = pe.Node(fsl.FEAT(), name='feat_fit', mem_gb=5)

# instead of FEAT
# modelestimate = pe.MapNode(interface=fsl.FILMGLS(smooth_autocorr=True,
#                                                 mask_size=5,
#                                                 threshold=1000),
#                                                 name='modelestimate',
#                                                 iterfield = ['design_file',
#                                                               'in_file',
#                                                               'tcon_file'])

feat_select = pe.Node(nio.SelectFiles({
    'cope': 'stats/cope*.nii.gz',
    'pe': 'stats/pe[0-9][0-9].nii.gz',
    'tstat': 'stats/tstat*.nii.gz',
    'varcope': 'stats/varcope*.nii.gz',
    'zstat': 'stats/zstat*.nii.gz',
}), name='feat_select')

ds_cope = pe.Node(DerivativesDataSink(
    base_directory=str(output_dir), keep_dtype=False, suffix='cope',
    desc='intask'), name='ds_cope', run_without_submitting=True)

ds_varcope = pe.Node(DerivativesDataSink(
    base_directory=str(output_dir), keep_dtype=False, suffix='varcope',
    desc='intask'), name='ds_varcope', run_without_submitting=True)

ds_zstat = pe.Node(DerivativesDataSink(
    base_directory=str(output_dir), keep_dtype=False, suffix='zstat',
    desc='intask'), name='ds_zstat', run_without_submitting=True)

ds_tstat = pe.Node(DerivativesDataSink(
    base_directory=str(output_dir), keep_dtype=False, suffix='tstat',
    desc='intask'), name='ds_tstat', run_without_submitting=True)

#%% connect workflow
workflow.connect([
    (infosource, selectfiles, [('subject_id', 'subject_id'), ('task_id', 'task_id')]),
    (selectfiles, runinfo, [('events','events_file'),('regressors','regressors_file')]),
    (selectfiles, susan, [('func', 'inputnode.in_files'), ('mask','inputnode.mask_file')]),
    (susan, runinfo, [('outputnode.smoothed_files', 'in_file')]),
    (susan, l1_spec, [('outputnode.smoothed_files', 'functional_runs')]),
  #  (susan,modelestimate, [('outputnode.smoothed_files','in_file')]), # try to run FILMGLS
    (selectfiles, ds_cope, [('func', 'source_file')]),
    (selectfiles, ds_varcope, [('func', 'source_file')]),
    (selectfiles, ds_zstat, [('func', 'source_file')]),
    (selectfiles, ds_tstat, [('func', 'source_file')]),
   
    (runinfo, l1_spec, [
        ('info', 'subject_info'),
        ('realign_file', 'realignment_parameters')]),
    (l1_spec, l1_model, [('session_info', 'session_info')]),
    (l1_model, feat_spec, [
        ('fsf_files', 'fsf_file'),
        ('ev_files', 'ev_files')]),
    (l1_model, feat_fit, [('fsf_files', 'fsf_file')]),
    # (feat_spec,modelestimate,[('design_file','design_file'),
    #                         ('con_file','tcon_file')]),
   
    (feat_fit, feat_select, [('feat_dir', 'base_directory')]),
    (feat_select, ds_cope, [('cope', 'in_file')]),
    (feat_select, ds_varcope, [('varcope', 'in_file')]),
    (feat_select, ds_zstat, [('zstat', 'in_file')]),
    (feat_select, ds_tstat, [('tstat', 'in_file')]),
])
    
#%% run workflow    
workflow.run(plugin='Linear', plugin_args={'n_procs': 1}) # try that in case fsl will run faster with it.
# workflow.run('MultiProc', plugin_args={'n_procs': 4,'memory_gb':40})