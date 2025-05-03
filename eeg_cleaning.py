import mne, os
from pathlib import Path
#from functions import make_directory
from itertools import product, batched
from joblib import Parallel, delayed
from time import ctime

def make_directory(fname):
    try:
        os.mkdir(fname)
    except FileExistsError:
        pass


############### Caution ###########################
## This program uses parallel processing <nj>    ##
###################################################
nj = 7

def load_eeg(s, se, c):                  
    # get the data path
    src = Path(rf'../CODE/MATB-data/sub-{s}/ses-S{se}')
                   
    if os.path.isdir(src) == False:
        print("PATH NOT FOUND!", src)
        raise SystemExit(1)
    
    # read the raw EEGLAB file (.set/.fdt) into MNE raw object
    raw = mne.io.read_raw_eeglab(src.joinpath(f'MATB{c}.set'), preload=True,
                                 eog=['HEOG', 'VEOG'])

    # not all the subjects have this channel, so just removing it
    raw = raw.drop_channels(['Cz'], on_missing='ignore')

    #  Set the montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, verbose=False)

    return raw

def clean_eeg(raw, use_ica):
    # step 1. apply a low-pass FIR filter
    raw_filtered = raw.filter(l_freq=1, h_freq=None, verbose=None, method='fir')

    # step 2. set common average reference and do re-referencing
    raw_ref = raw_filtered.set_eeg_reference("average", projection=False, 
                                             verbose=False, ch_type = 'eeg')
    
    # Two versions tested - clean (with ICA) and 'less clean' (no ICA)
    if (use_ica == True):
        
        # Step 3 apply Independent Componenet Analysis (ICA)
        ica = mne.preprocessing.ICA(n_components=len(raw.ch_names)-2,
                                    method = 'picard', max_iter="auto", 
                                    random_state=42,
                                    fit_params=dict(ortho=True, extended=True))
        
        # Run the ICA decomposition on the re-referenced data
        ica.fit(raw_ref)
        ica.exclude = []

        ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG1', method="correlation", 
                                                    threshold="auto", verbose=0)
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='Fp1', verbose=0,)
        muscle_indices, muscle_scores = ica.find_bads_muscle(raw, verbose=0,)

        # Add the indices to be removed.
        ica.exclude.extend(eog_indices)
        ica.exclude.extend(ecg_indices)
        ica.exclude.extend(muscle_indices)
        
        print("ICA Results:")
        print(f"{len(eog_indices)} comps removed due to eye artefact.")
        print(f"{len(ecg_indices)} comps removed due to heart artefact.")
        print(f"{len(muscle_indices)} comps removed due to muscle artefact.")

        ica.apply(raw_ref, exclude=ica.exclude)

    # 12 seconds fixed segments 
    start, dur = 0, 12
    new_events = mne.make_fixed_length_events(raw, start=start, stop=None, 
                                              duration=dur, overlap=0)
    
    # form EEG segmnets of SIX SECONDS EACH (NO baseline correction)
    epochs = mne.Epochs(raw_ref, new_events, tmin=start, tmax=dur, 
                        baseline=None, verbose=0, reject_by_annotation=True,
                        preload=True, reject=None, flat=None)
    #epochs.plot(n_epochs=1, block=True)
    print(epochs.__len__())
    return epochs # Return the cleaned data.

def run_cogbci(path):
    n_subjects, n_sessions = 29, 3
    sessions = list(range(1, n_sessions+1))
    subjects = list(range(1, n_subjects+1))
    ica_control = [(True, ''), (False, '_noica')]
    levels = [('easy', 'easy'), ('diff', 'hard')]
    lst = list(product(subjects, sessions, ica_control, levels))
    groups =  list(batched(lst, nj))
    print("There are", len(lst), "EEG files to process, and", len(groups), "groups.")

    def f(n, se, ica_info, level_info):
        s = str(n).zfill(2)
        level, _ = level_info
        r = load_eeg(s, se, level)
        
        if r != None: # no eye/heart artefact removal or resample
            ica_flag = ica_info[0]
            return clean_eeg(r, ica_flag)
        return r

    counter = 0
    for group in groups:
        print("************** group", counter, ctime(), "*******************")
        print('parallels started', ctime(), flush=True)
        output = Parallel(n_jobs=nj, verbose=1)(delayed(f)(*t) for t in group)
        print('parallels finished', ctime(), flush=True)
        
        # the output will be in sync w/order of the input group.
        for t, e in zip(group, output): #t: lst tuple, e: epochs output
            n, se, ica_info, level_info = t
            _, flg = ica_info
            _, label = level_info
            
            s = str(n).zfill(2)
            p = f"{path}/{s}/ses-{se}"
            make_directory(f"{path}/{s}")
            make_directory(p)
            
            if e != None:
                e.save(f'{p}/epochs_{label}{flg}_epo.fif.gz', overwrite=True)

        counter += 1

def main():
    path = 'clean_eeg'
    make_directory(path)
    run_cogbci(path)

if __name__ == '__main__':
    main()
    