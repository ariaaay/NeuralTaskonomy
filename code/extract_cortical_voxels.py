"""
This scripts extracts cortical voxels from proprocessed brain volume data using
precomputed brain mask from Pycortex.
"""


import cortex
import nibabel as nib
import numpy as np
import os
import pickle
import csv
import json

from util.util import *

bpath = os.environ["BOLD5000"]
out_dir = "/media/tarrlab/scenedata2/BOLD5000_cortical"

db = cortex.database.default_filestore

TROI = [3, 4] #TR of interest

def initiate_subject(subj):
    if "sub-CSI{}".format(subj) not in cortex.db.subjects:
        print(
            "Subjects {} data does not exist in pycortex database. Initiating..".format(
                subj
            )
        )
        # initiate subjects
        # if this returns Key Error, manually enter the following lines in ipython works
        cortex.freesurfer.import_subj("sub-CSI" + str(subj))

    transform_name = "full"
    transform_path = "{}/sub-CSI{}/transforms/{}".format(db, subj, transform_name)

    if not os.path.isdir(transform_path):  # no transform generated yet
        print("No transform found. Auto aligning...")

        # load a reference slice for alignment'/data2/tarrlab/common/datasets/pycortex_db/sub-CSI{}/func_examples/
        slice_dir = "{}/sub-CSI{}/func_examples/".format(db, subj)
        if not os.path.isdir(slice_dir):
            os.makedirs(slice_dir)
        slice_path = slice_dir + "slice.nii.gz"

        try:
            nib.load(slice_path)
        except FileNotFoundError:
            sample_run = (
                "{}/derivatives/fmriprep/sub-CSI{}/ses-01/func/sub-CSI{}_ses-01_task-5000scenes_"
                "run-01_bold_space-T1w_preproc.nii.gz".format(bpath, subj, subj)
            )
            img = nib.load(sample_run)
            d = img.get_data()
            dmean = np.mean(d, axis=3)
            sample_slice = nib.Nifti1Image(dmean, img.affine)
            nib.save(sample_slice, slice_path)
        # run automatic alignment
        cortex.align.automatic("sub-CSI" + str(subj), transform_name, slice_path)
        # creates a reference transform matrix for this functional run in filestore/db/<subject>/transforms


def load_TR(data_path, stimuli_path, mask, subj, TR=3, ISP=5):
    """
    Convert serialized pickle file into numpy matrix, run by run.
    :param data_path: path of the whole brain data
    :param stimuli_path: path of the stimuli presentation file
    :param TR: the TR of interest in every stimuli presentations
    :param ISP: Interval of stimuli  presentation, in terms of TR (default = 10s)
    :return data_mat_TR: data matrix of the extracted TR
    :return stimuli_list: list of images that were presented
    :return RT: reaction time
    :return Valence: valence to the image

    """
    stimuli_list, valence_list, RT_list, sess_list, run_list, trial_list = (
        list(),
        list(),
        list(),
        list(),
        list(),
        list(),
    )
    with open(stimuli_path, "r") as f:
        events = csv.reader(f, delimiter="\t")
        next(events)  # skip header
        for r, row in enumerate(events):
            stimuli_list.append(row[-1])
            RT_list.append(row[11])
            valence_list.append(row[10])
            sess_list.append(row[3])
            run_list.append(row[4])
            trial_list.append(row[5])
            if r == 0:
                I1 = int(
                    float(row[0]) / 2
                )  # the presentations of first stimuli, in terms of TR (6s)

    data_mat = list()
    img = nib.load(data_path)
    d = img.get_data()
    num_scans = d.shape[-1]
    sample_ind = np.arange(I1 + TR, num_scans, ISP)[
        : len(stimuli_list)
    ]  # ignore scans with no image if there's any
    for t in sample_ind:
        dt = d[..., t]
        dt = dt.T
        # print(d.shape)
        # print(mask.shape)
        cortical_data = dt[mask]
        data_mat.append(cortical_data)
    data_mat = np.array(data_mat)
    assert data_mat.shape[0] == len(stimuli_list)
    return (
        data_mat,
        stimuli_list,
        RT_list,
        valence_list,
        sess_list,
        run_list,
        trial_list,
    )


def strn(num):
    "append 0 in front of a single digit number for loading files"
    if len(str(num)) == 1:
        run_name = "0" + str(num)
    else:
        run_name = str(num)
    return run_name


def reformat_data():
    for subj in np.arange(1, 2, 1):
        initiate_subject(subj)
        mask = cortex.utils.get_cortical_mask("sub-CSI{}".format(subj), "full")

        for TR in TROI:
            all_data = None
            stims, RTs, valences, sessions, runs, trials = (
                list(),
                list(),
                list(),
                list(),
                list(),
                list(),
            )
            for s in range(15):  # b/c not all subjects had the same session numbers
                s += 1
                r = 0
                data = None
                while True:
                    r += 1
                    print(
                        "Processing subject {}, session {}, run {}, for TR={}...".format(
                            subj, strn(s), strn(r), TR
                        )
                    )
                    try:
                        event_path = (
                            bpath
                            + "/sub-CSI{}/ses-{}/func/sub-CSI{}_ses-{}_task-5000scenes_run-{}_events.tsv".format(
                                subj, strn(s), subj, strn(s), strn(r)
                            )
                        )
                        brain_path = (
                            bpath
                            + "/derivatives/fmriprep/sub-CSI{}/ses-{}/func/sub-CSI{}_ses-{}_task-5000scenes_run-{}_bold_space-T1w_preproc.nii.gz".format(
                                subj, strn(s), subj, strn(s), strn(r)
                            )
                        )
                        d, stim, RT, valence, session, run, trial = load_TR(
                            brain_path, event_path, mask, subj, TR=TR
                        )
                        stims.extend(stim)
                        RTs.extend(RT)
                        valences.extend(valence)
                        sessions.extend(session)
                        runs.extend(run)
                        trials.extend(trial)
                        d = zscore(d, axis=0)
                        if data is None:
                            data = d
                        else:
                            data = np.vstack((data, d))
                    except FileNotFoundError:
                        break

                if all_data is None:
                    all_data = data
                else:
                    all_data = np.vstack((all_data, data))

            np.save(out_dir + "/CSI{}_TR{}_zscore.npy".format(subj, TR), all_data)
            event_dict = dict()
            event_dict["stims"] = stims
            event_dict["RT"] = RTs
            event_dict["valence"] = valences
            event_dict["session"] = sessions
            event_dict["runs"] = runs
            event_dict["trials"] = trials
            with open(out_dir + "/CSI{}_events.json".format(subj), "w") as f:
                json.dump(event_dict, f)


if __name__ == "__main__":
    reformat_data()
