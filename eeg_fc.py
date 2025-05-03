import numpy as np
from time import ctime
from joblib import Parallel, delayed
from functions import *

##################################### WARNING ################################
## THIS PROGRAM USES PARALLEL COMPUTATIONS. IF YOUR COMPUTER DOES NOT HAVE  ##
## ENOUGH MEMORY, REDUCE THE NUMBER OF PARALLEL PROCESSES (nj) ###########
##############################################################################

############################## INFORMATION ##################################
## The list pair_data is used to store pairs which generate Figure 7.
#############################################################################

nj = 7
n_subjects, n_sessions = 29, 3
gtype = 'HVG'
labels = ['CSS', 'NMI', 'ALE', 'IC', 'MSC', 'WPLI', 'PCC']
levels = ['easy', 'hard']
band_names, bands = ['theta'], [[4, 7]]

def g(index, se, use_ica):
    global bands, band_names, labels, levels

    s = str(index + 1).zfill(2)
    n_bands, n_methods, n_levels = len(bands), len(labels), len(levels)
    
    ch_names=['F3','Fz','F4','P3','Pz','P4']
    nchan = len(ch_names)
    res = [[[[] for _ in range(n_methods)] for _ in range(n_bands)] \
           for _ in range(n_levels)]
    pair_data = [[[] for _ in range(nchan)] for _ in range(n_levels)]
    
    lev_ct = 0
    for level in levels:
        p = f"clean_eeg/{s}/ses-{se}"
        
        if use_ica:
            epochs = mne.read_epochs(f'{p}/epochs_{level}_epo.fif.gz')
        else:
            epochs = mne.read_epochs(f'{p}/epochs_{level}_noica_epo.fif.gz')
            
        fs = int(epochs.info.get('sfreq'))

        iterator = epochs.get_data(picks=ch_names)
        nchan, n_epochs = len(ch_names), len(iterator)
        
        band_ct = 0
        for band_name, band in zip(band_names, bands):
            print('starting sub', s, 'se', se, 'level', level, 'band', band_name,
                  'at', ctime(), flush=True)
            mean_matrices = [[] for _ in range(n_methods)]
            
            c = 0
            for data in iterator:
                # Tracks which EEG channels have already added to pair_data
                pairs_added = [] 
                if 1 == 1:  # was used for testing, can be ignored
                    matrices = [np.zeros((nchan, nchan)) for _ in range(n_methods)]                  
                    matrix_ct = 0 # global counter index of the matrix
                    for j in range(nchan):
                        for k in range(nchan):
                            if (j <= k): # all SPIs are symmetric - no directionality

                                x, y = data[j], data[k]
                                labels, scores, x_pairs, y_pairs = eeg_fc(fs, x, y, gtype, band)

                                if j not in pairs_added:
                                    pair_data[lev_ct][j].extend(x_pairs)
                                    pairs_added.append(j)

                                if k not in pairs_added:
                                    pair_data[lev_ct][k].extend(y_pairs)
                                    pairs_added.append(k)
                                
                                for score, fc_ct in zip(scores, range(len(scores))):
                                    matrices[fc_ct][j][k] = score
                                    matrices[fc_ct][k][j] = score                                        
                            
                                matrix_ct += 1
                            
                    for mx, method_ct in zip(matrices, range(n_methods)):
                        mean_matrices[method_ct].append(mx)

                c += 1

            method_ct = 0  # AVERAGE OF ALL EPOCHS
            for matrix_lst, method in zip(mean_matrices, labels):
                mean_matrix = np.mean(matrix_lst, axis=0)
                res[lev_ct][band_ct][method_ct].append(mean_matrix)
                method_ct += 1
        
            band_ct += 1
        lev_ct += 1
    
    return (res, pair_data)

folder = 'EEG_SPI_results'
make_directory(folder)

# Splitting into groups to handle the memory cost of this program running.
groups = np.array_split(range(8, n_subjects), 3)
for group in groups:
    print("group", group, ctime())
    for se in range(1, n_sessions+1):
        lst = (list(map(lambda x: (x, se), group)))

        f1, f2 = ['', '_noica'], [True, False]
        for ica_flag, use_ica in zip(f1, f2):
            print('parallels started', ctime(), flush=True)
            output = Parallel(n_jobs=nj, verbose=2)(delayed(g)(s_ind, se_ind, use_ica \
                     ) for (s_ind, se_ind) in lst)
            print('parallels done', ctime(), flush=True)

            lev_ct = 0
            for level in levels:
                print("saving", level)
                make_directory(f'{folder}/{level}')
                
                band_ct = 0
                for band_name in band_names:
                    make_directory(f'{folder}/{level}/{band_name}')
                    
                    method_ct = 0
                    for method in labels:
                        save_path = f'{folder}/{level}/{band_name}/{method}'
                        make_directory(f'{folder}/{level}/{band_name}/{method}')
                        
                        subj_index = group[0]
                        for r in output:
                            s = str(subj_index + 1).zfill(2)
                            matrices = r[0][lev_ct][band_ct][method_ct]
                            
                            for matrix in matrices:
                                q = f'{save_path}/sub{s}_se{se}{ica_flag}.csv'
                                np.savetxt(q, matrix, delimiter=',')
                            
                            subj_index += 1

                        method_ct += 1
                    band_ct += 1
                lev_ct += 1
                
            # this part saves the pair_data
            n_pairs = len(output[0][1][0])
            lev_ct = 0
            for level in levels:
                print("saving", level)
                make_directory(f'{folder}/{level}/pair_data')

                subj_index = group[0]
                for r in output:
                    s = str(subj_index + 1).zfill(2)
                    
                    # shape needs to be (2 levels, 21 FC pairs, N_Segments)
                    for pair in range(n_pairs):
                        data = np.array(r[1][lev_ct][pair])
                        
                        save_path = f'{folder}/{level}/pair_data'
                        q = f'{save_path}/pair{pair}_sub{s}_se{se}{ica_flag}.csv'
                        np.savetxt(q, data, delimiter=',') 
                    
                    subj_index += 1
                        
                lev_ct += 1
        
        del output
