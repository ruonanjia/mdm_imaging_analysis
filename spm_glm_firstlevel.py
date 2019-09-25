# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
import numpy as np
#%%
base_root = '/home/rj299/project/mdm_analysis/'
data_root = '/home/rj299/project/mdm_analysis/data_rename'
out_root = '/home/rj299/project/mdm_analysis/output'

# base_root = 'D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Imaging Analysis\RA_PTSD_SPM'
# data_root = 'D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Imaging Analysis\data_rename'
# out_root = 'D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Imaging Analysis\output'

#%%
from nipype.interfaces import spm

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
#import nipype.algorithms.rapidart as ra  # artifact detection
import nipype.algorithms.modelgen as model  # model specification
#from nipype.algorithms.rapidart import ArtifactDetect
# from nipype.algorithms.misc import Gunzip
from nipype import Node, Workflow, MapNode
from nipype.interfaces import fsl

from nipype.interfaces.matlab import MatlabCommand

#%%
MatlabCommand.set_default_paths('/home/rj299/project/MATLAB/toolbox/spm12/') # set default SPM12 path in my computer. 
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

data_dir = data_root
output_dir = os.path.join(out_root, 'imaging')
work_dir = os.path.join(base_root, 'work') # intermediate products

# subject_list = [2583, 2588]
# task_list = [1,2,3,4,5,6,7,8]

#subject_list = [2588]
#subject_list = [2073, 2550, 2582, 2583, 2584, 2585]
subject_list = [2582, 2583, 2584]
# task_id = [1,2]

fwhm = 6
tr = 1
# first sevetal scans to delete
del_scan = 10

# Map field names to individual subject runs.
# infosource = pe.Node(util.IdentityInterface(fields=['subject_id', 'task_id'],),
#                   name="infosource")

# infosource.iterables = [('subject_id', subject_list), 
#                         ('task_id', task_list)]

infosource = pe.Node(util.IdentityInterface(fields=['subject_id'],),
                  name="infosource")

infosource.iterables = [('subject_id', subject_list)]


#%%
def _bids2nipypeinfo(in_file, events_file, regressors_file,
                     regressors_names=None,
                     motion_columns=None,
                     decimals=3, amplitude=1.0, del_scan=10):
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
    np.savetxt(out_motion, regress_data[motion_columns].fillna(0.0).values[del_scan:,], '%g')
#     np.savetxt(out_motion, regress_data[motion_columns].fillna(0.0).values, '%g')
    
    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))

    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']

    domain = list(set(events.condition.values))[0] # domain of this task run, should be only one, 'Mon' or 'Med'
    trial_types = list(set(events.trial_type.values))
 
    runinfo = Bunch(
        scans=in_file,
        conditions=[domain + '_' + trial_type for trial_type in trial_types],
        # conditions = ['Med_amb', 'Med_risk', 'Mon_amb', 'Mon_risk'],
        **{k: [] for k in bunch_fields})

    for condition in runinfo.conditions:        
        
        event = events[events.trial_type.str.match(condition[4:])]
        runinfo.onsets.append(np.round(event.onset.values - del_scan + 1, 3).tolist()) # take out the first several deleted scans
        runinfo.durations.append(np.round(event.duration.values, 3).tolist())
        if 'amplitudes' in events.columns:
            runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
        else:
            runinfo.amplitudes.append([amplitude] * len(event))
            
        # if domain == condition[:3]:
        #     event = events[events.trial_type.str.match(condition[4:])]
        #     runinfo.onsets.append(np.round(event.onset.values - del_scan + 1, 3).tolist()) # take out the first several deleted scans
        #     runinfo.durations.append(np.round(event.duration.values, 3).tolist())
        #     if 'amplitudes' in events.columns:
        #         runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
        #     else:
        #         runinfo.amplitudes.append([amplitude] * len(event))
                
        # else: # empty conditions
        #     runinfo.onsets.append([])
        #     runinfo.durations.append([])
        #     runinfo.amplitudes.append([])
            
            
    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        runinfo.regressors = regress_data[regressors_names].fillna(0.0).values[del_scan:,].T.tolist()

    return runinfo, str(out_motion)

#%%
templates = {'func': os.path.join(data_root, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-{task_id}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
             'mask': os.path.join(data_root, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-{task_id}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'),
             'regressors': os.path.join(data_root, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-{task_id}_desc-confounds_regressors.tsv'),
             'events': os.path.join(out_root, 'event_files', 'sub-{subject_id}_task-{task_id}_cond.csv')}

# Flexibly collect data from disk to feed into workflows.
selectfiles = pe.Node(nio.SelectFiles(templates,
                      base_directory=data_root),
                      name="selectfiles")
        
selectfiles.inputs.task_id = [1,2,3,4,5,6,7,8]        
        
# Extract motion parameters from regressors file
runinfo = MapNode(util.Function(
    input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names', 'motion_columns'],
    function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
    name='runinfo',
    iterfield = ['in_file', 'events_file', 'regressors_file'])

# Set the column names to be used from the confounds file
# reference a paper from podlrack lab
runinfo.inputs.regressors_names = ['std_dvars', 'framewise_displacement'] + \
                                  ['a_comp_cor_%02d' % i for i in range(6)]
                                  

runinfo.inputs.motion_columns   = ['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2'] + \
                                  ['trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2'] + \
                                  ['trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2'] + \
                                  ['rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2'] + \
                                  ['rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2'] + \
                                  ['rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']


#%%
# gunzip = MapNode(Gunzip(), name='gunzip', iterfield=['in_file'])


# delete first several scans
# def extract_all(in_files):
#     from nipype.interfaces import fsl
#     roi_files = []
#     for in_file in in_files:
#         roi_file = fsl.ExtractROI(in_file = in_file, t_min = 10, t_size = -1, output_type = 'NIFTI')
#         roi_files.append(roi_file)  
#     return roi_files

        
# extract = pe.Node(util.Function(
#         input_names = ['in_files'],
#         function = extract_all, output_names = ['roi_files']),
#         name = 'extract')
        
extract = pe.MapNode(fsl.ExtractROI(), name="extract", iterfield = ['in_file'])
extract.inputs.t_min = del_scan
extract.inputs.t_size = -1
extract.inputs.output_type='NIFTI'

# smoothing
smooth = Node(spm.Smooth(), name="smooth", fwhm = fwhm)

# set contrasts, depend on the condition
cont1 = ['Med_Amb', 'T', ['Med_amb', 'Med_risk', 'Mon_amb', 'Mon_risk'], [1, 0, 0, 0]]
cont2 = ['Med_Risk', 'T', ['Med_amb', 'Med_risk', 'Mon_amb', 'Mon_risk'], [0, 1, 0, 0]]
cont3 = ['Med_Amb>Risk', 'T', ['Med_amb', 'Med_risk', 'Mon_amb', 'Mon_risk'], [1, -1, 0, 0]]

cont4 = ['Mon_Amb', 'T', ['Med_amb', 'Med_risk', 'Mon_amb', 'Mon_risk'], [0, 0, 1, 0]]
cont5 = ['Mon_Risk', 'T', ['Med_amb', 'Med_risk', 'Mon_amb', 'Mon_risk'], [0, 0, 0, 1]]
cont6 = ['Mon_Amb>Risk', 'T', ['Med_amb', 'Med_risk', 'Mon_amb', 'Mon_risk'], [0, 0, 1, -1]]

cont7 = ['Med>Mon_Amb', 'T', ['Med_amb', 'Med_risk', 'Mon_amb', 'Mon_risk'], [1, 0, -1, 0]]
cont8 = ['Med>Mon_Risk', 'T', ['Med_amb', 'Med_risk', 'Mon_amb', 'Mon_risk'], [0, 1, 0, -1]]

cont9 = ['Med>Mon', 'T', ['Med_amb', 'Med_risk', 'Mon_amb', 'Mon_risk'], [1, 1, -1, -1]]

contrasts = [cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9]

# cont1 = ['Med_Amb', 'T', ['Med_amb', 'Med_risk'], [1, 0]]
# cont2 = ['Med_Risk', 'T', ['Med_amb', 'Med_risk'], [0, 1]]
# contrasts = [cont1, cont2]

#%%

modelspec = Node(interface=model.SpecifySPMModel(), name="modelspec") 
modelspec.inputs.concatenate_runs = False
modelspec.inputs.input_units = 'scans' # supposedly it means tr
modelspec.inputs.output_units = 'scans'
#modelspec.inputs.outlier_files = '/media/Data/R_A_PTSD/preproccess_data/sub-1063_ses-01_task-3_bold_outliers.txt'
modelspec.inputs.time_repetition = 1.  # make sure its with a dot 
modelspec.inputs.high_pass_filter_cutoff = 128.

level1design = pe.Node(interface=spm.Level1Design(), name="level1design") #, base_dir = '/media/Data/work')
level1design.inputs.timing_units = modelspec.inputs.output_units
level1design.inputs.interscan_interval = 1.
level1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
level1design.inputs.model_serial_correlations = 'AR(1)'

# create workflow
wfSPM = Workflow(name="l1spm", base_dir=work_dir)
wfSPM.connect([
        (infosource, selectfiles, [('subject_id', 'subject_id')]),
        (selectfiles, runinfo, [('events','events_file'),('regressors','regressors_file')]),
        (selectfiles, extract, [('func','in_file')]),
        (extract, smooth, [('roi_file','in_files')]),
        (smooth, runinfo, [('smoothed_files','in_file')]),
        (smooth, modelspec, [('smoothed_files', 'functional_runs')]),   
        (runinfo, modelspec, [('info', 'subject_info'), ('realign_file', 'realignment_parameters')]),
        
        ])
wfSPM.connect([(modelspec, level1design, [("session_info", "session_info")])])

#%%
level1estimate = pe.Node(interface=spm.EstimateModel(), name="level1estimate")
level1estimate.inputs.estimation_method = {'Classical': 1}

contrastestimate = pe.Node(
    interface=spm.EstimateContrast(), name="contrastestimate")
#contrastestimate.inputs.contrasts = contrasts
contrastestimate.overwrite = True
contrastestimate.config = {'execution': {'remove_unnecessary_outputs': False}}
contrastestimate.inputs.contrasts = contrasts                                                   
                                                   

wfSPM.connect([
         (level1design, level1estimate, [('spm_mat_file','spm_mat_file')]),
         (level1estimate, contrastestimate,
            [('spm_mat_file', 'spm_mat_file'), ('beta_images', 'beta_images'),
            ('residual_image', 'residual_image')]),
    ])

#%% Adding data sink
########################################################################
# Datasink
datasink = Node(nio.DataSink(base_directory=os.path.join(output_dir, 'Sink')),
                                         name="datasink")
                       

wfSPM.connect([
       # here we take only the contrast ad spm.mat files of each subject and put it in different folder. It is more convenient like that. 
       (contrastestimate, datasink, [('spm_mat_file', '1stLevel.@spm_mat'),
                                              ('spmT_images', '1stLevel.@T'),
                                              ('con_images', '1stLevel.@con'),
                                              ('spmF_images', '1stLevel.@F'),
                                              ('ess_images', '1stLevel.@ess'),
                                              ])
        ])

#%% run
wfSPM.run('MultiProc', plugin_args={'n_procs': 4})
    
#%%
# wfSPM.write_graph(graph2use = 'flat')

# # wfSPM.write_graph("workflow_graph.dot", graph2use='colored', format='png', simple_form=True)
# # wfSPM.write_graph(graph2use='orig', dotfilename='./graph_orig.dot')
# %matplotlib inline
# from IPython.display import Image
# %matplotlib qt
# Image(filename = '/home/rj299/project/mdm_analysis/work/l1spm/graph.png')
# #%% FSL    
                                                   
## 
#l1_spec = pe.Node(SpecifyModel(
#    parameter_source='FSL',
#    input_units='secs',
#    high_pass_filter_cutoff=120,
#    time_repetition = tr,
#), name='l1_spec')
#
## l1_model creates a first-level model design
#l1_model = pe.Node(fsl.Level1Design(
#    bases={'dgamma': {'derivs': True}}, # adding temporal derivative of double gamma
#    model_serial_correlations=True,
#    interscan_interval = tr,
#    contrasts=contrasts
#    # orthogonalization=orthogonality,
#), name='l1_model')
#
## feat_spec generates an fsf model specification file
#feat_spec = pe.Node(fsl.FEATModel(), name='feat_spec')
#
## feat_fit actually runs FEAT
#feat_fit = pe.Node(fsl.FEAT(), name='feat_fit', mem_gb=5)
#
### instead of FEAT
##modelestimate = pe.MapNode(interface=fsl.FILMGLS(smooth_autocorr=True,
##                                                 mask_size=5,
##                                                 threshold=1000),
##                                                 name='modelestimate',
##                                                 iterfield = ['design_file',
##                                                              'in_file',
##                                                              'tcon_file'])
#
#feat_select = pe.Node(nio.SelectFiles({
#    'cope': 'stats/cope*.nii.gz',
#    'pe': 'stats/pe[0-9][0-9].nii.gz',
#    'tstat': 'stats/tstat*.nii.gz',
#    'varcope': 'stats/varcope*.nii.gz',
#    'zstat': 'stats/zstat*.nii.gz',
#}), name='feat_select')
#
#ds_cope = pe.Node(DerivativesDataSink(
#    base_directory=str(output_dir), keep_dtype=False, suffix='cope',
#    desc='intask'), name='ds_cope', run_without_submitting=True)
#
#ds_varcope = pe.Node(DerivativesDataSink(
#    base_directory=str(output_dir), keep_dtype=False, suffix='varcope',
#    desc='intask'), name='ds_varcope', run_without_submitting=True)
#
#ds_zstat = pe.Node(DerivativesDataSink(
#    base_directory=str(output_dir), keep_dtype=False, suffix='zstat',
#    desc='intask'), name='ds_zstat', run_without_submitting=True)
#
#ds_tstat = pe.Node(DerivativesDataSink(
#    base_directory=str(output_dir), keep_dtype=False, suffix='tstat',
#    desc='intask'), name='ds_tstat', run_without_submitting=True)

#%% connect workflow, FSL
#workflow.connect([
#    (infosource, selectfiles, [('subject_id', 'subject_id'), ('task_id', 'task_id')]),
#    (selectfiles, runinfo, [('events','events_file'),('regressors','regressors_file')]),
#    (selectfiles, susan, [('func', 'inputnode.in_files'), ('mask','inputnode.mask_file')]),
#    (susan, runinfo, [('outputnode.smoothed_files', 'in_file')]),
#    (susan, l1_spec, [('outputnode.smoothed_files', 'functional_runs')]),
#  #  (susan,modelestimate, [('outputnode.smoothed_files','in_file')]), # try to run FILMGLS
#    (selectfiles, ds_cope, [('func', 'source_file')]),
#    (selectfiles, ds_varcope, [('func', 'source_file')]),
#    (selectfiles, ds_zstat, [('func', 'source_file')]),
#    (selectfiles, ds_tstat, [('func', 'source_file')]),
#   
#    (runinfo, l1_spec, [
#        ('info', 'subject_info'),
#        ('realign_file', 'realignment_parameters')]),
#    (l1_spec, l1_model, [('session_info', 'session_info')]),
#    (l1_model, feat_spec, [
#        ('fsf_files', 'fsf_file'),
#        ('ev_files', 'ev_files')]),
#    (l1_model, feat_fit, [('fsf_files', 'fsf_file')]),
##    (feat_spec,modelestimate,[('design_file','design_file'),
##                            ('con_file','tcon_file')]),
#   
#    (feat_fit, feat_select, [('feat_dir', 'base_directory')]),
#    (feat_select, ds_cope, [('cope', 'in_file')]),
#    (feat_select, ds_varcope, [('varcope', 'in_file')]),
#    (feat_select, ds_zstat, [('zstat', 'in_file')]),
#    (feat_select, ds_tstat, [('tstat', 'in_file')]),
#])
#    
#%% run workflow, FSL    
#workflow.run(plugin='Linear', plugin_args={'n_procs': 1}) # try that in case fsl will run faster with it.
# workflow.run('MultiProc', plugin_args={'n_procs': 4,'memory_gb':40})