import numpy as np
from ts2vg import HorizontalVG, NaturalVG
import mne
from sklearn.metrics import normalized_mutual_info_score
from collections import OrderedDict
from scipy.signal import correlate, correlation_lags, butter, sosfilt
from scipy.stats import pearsonr
from scipy.signal import csd
import colorednoise as cn
from statsmodels.stats.dist_dependence_measures import distance_statistics
from time import ctime
from scipy.stats import wilcoxon, ttest_rel
from scipy.stats import sem
from collections import Counter   
import igraph as ig
import leidenalg as la
from os import mkdir
from scipy.optimize import curve_fit

def make_directory(fname):
    try:
        mkdir(fname)
    except FileExistsError:
        pass

def get_css_sequence(x, w, w_s, centrality_method):
    n_points, s_x, pair_data = len(x), [], []
    i = 0
    while (i + w <= n_points):
        e_i = range(i, i + w)
        G = gen_vg([x[e_i]], 'HVG')[0]
        
        comms = list(la.find_partition(G, la.ModularityVertexPartition, 
                                       weights='weight', seed=5))
        csm_list = []
        for ct in range(len(comms)):
            comm = G.subgraph(comms[ct])
            if centrality_method == 'closeness':
                csm_vals = comm.closeness(weights='weight')
            elif centrality_method == 'betweenness':
                csm_vals = comm.betweenness(weights='weight')
            elif centrality_method == 'eigenvector':
                csm_vals = comm.eigenvector_centrality(weights='weight')
            
            avg_csm_val = np.mean(csm_vals)
            csm_list.append(avg_csm_val)

            avg_wt_degree = np.mean(comm.strength(weights='weight'))
            pair_data.append((avg_wt_degree, avg_csm_val))
            
        s_x.append(np.mean(csm_list))
        i = (i + w) - w_s
        
    return (np.array(s_x), pair_data)
    
def css(x, y, w, w_s, centrality_method):
    s_x, x_pair_data = get_css_sequence(x, w, w_s, centrality_method)
    s_y, y_pair_data = get_css_sequence(y, w, w_s, centrality_method)
    css_score = distance_statistics(s_x, s_y).distance_correlation
    return (css_score, x_pair_data, y_pair_data)

# Layer Entanglement / Edge overlap method using igraph's ecount()/intersection() methods
def edge_overlap(g1, g2):
    res = g1.intersection(g2)
    count = res.ecount()
    g1_score = count / g1.ecount()
    g2_score = count / g2.ecount()
    avg_score = np.mean([g1_score, g2_score])
    return (g1_score, g2_score, avg_score)

# Normalized mutual information using Scikit-Learn function
def get_final_corr(graphs):
    counts = Counter(list(graphs[0].degree()))
    a = [counts.get(i, 0) for i in range(max(counts) + 1 if counts else 0)]
    
    counts = Counter(list(graphs[1].degree()))
    b = [counts.get(i, 0) for i in range(max(counts) + 1 if counts else 0)]
    
    maxlen = max([len(a), len(b)])
    a = list(map(lambda i: a[i] if i < len(a) else 0, range(maxlen)))
    b = list(map(lambda i: b[i] if i < len(b) else 0, range(maxlen)))
    nmi = normalized_mutual_info_score(a,b)
    return nmi

# Generates the visibility graph
def gen_vg(data, gtype):
    if (gtype == 'HVG'):
        w = 'abs_slope'
        f = lambda i: (HorizontalVG(weighted=w, penetrable_limit=1).build(data[i])).as_igraph()
    else:
        print("invalid graph type")
        raise SystemExit(1)

    vg_list = list(map(f, range(len(data))))
    return vg_list

def eeg_fc(fs, x, y, gtype, band):
    # Using Welch's method to estimate the power spectral density
    # The choice of window is recommended to "encompass 2 cycles of lowest
    # frequency of interest" - for us it is 4 Hz, so 2/4 = 0.5 seconds
    nperseg = fs * 0.5 # 250 samples per segment.
    f_min, f_max = band[0], band[1]
    f, sxy = csd(x, y, fs=fs, nperseg=nperseg)
    f, sxx = csd(x, x, fs=fs, nperseg=nperseg)
    f, syy = csd(y, y, fs=fs, nperseg=nperseg)
    w = np.where((f>=f_min) & (f<=f_max))
    
    # Frequency domain methods
    msc = (np.abs(np.mean(sxy[w]))**2) / (np.mean(sxx[w]) * np.mean(syy[w]))
    wpli = np.abs(np.mean(np.imag(sxy[w]))) / np.mean(np.abs(np.imag(sxy[w])))
    ic = np.abs(np.imag(np.mean(sxy[w])) / np.sqrt(np.mean(sxx[w]) * np.mean(syy[w])))

    # Time/Network domain methods -- use a bandpass filter
    sos = butter(3, btype='bandpass', Wn=[f_min, f_max], fs=fs, output='sos')
    data = np.array([x, y])
    fil_data = np.array(list(map(lambda j: sosfilt(sos, data[j]), 
                                 range(len(data)))))
    vg_list = gen_vg(fil_data, gtype)
    nmi = get_final_corr(vg_list)
    pcc = np.abs(pearsonr(fil_data[0], fil_data[1]).statistic)
    _, _, avg_le = edge_overlap(vg_list[0], vg_list[1])

    omega = fs//2
    css_score, x_pairs, y_pairs = css(fil_data[0], fil_data[1], omega, 
                                      omega//2, 'closeness')
    
    labels = ['CSS', 'NMI', 'ALE', 'IC', 'MSC', 'WPLI', 'PCC']
    scores = [css_score, nmi, avg_le, ic, msc, wpli, pcc]
    return (labels, scores, x_pairs, y_pairs)

def get_data(fs, win, length, delay, beta, rstate):
    y1 = cn.powerlaw_psd_gaussian(beta, length + delay, random_state=rstate)
    sos = butter(3, win, btype='bandpass', output='sos', fs=fs)
    y1 = sosfilt(sos, y1)
    temp = y1
    y2 = y1[delay:]
    y1 = temp[:length]
    # no need to filter these because already did above
    y1, y2 = y1/np.linalg.norm(y1), y2/np.linalg.norm(y2)
    return (y1, y2)

def normalize_signal(signal, fs, win):
    sos = butter(3, win, btype='bandpass', output='sos', fs=fs)
    signal_filtered = sosfilt(sos, signal)
    norm = np.linalg.norm(signal_filtered)    
    return signal/norm

def gen_norm_noise(beta, length, fs, win):
    n = cn.powerlaw_psd_gaussian(beta, length, random_state=None)
    return normalize_signal(n, fs, win)

def get_noise(theta, length, fs, win):
    w1 = gen_norm_noise(0, length, fs, win)
    p1 = gen_norm_noise(1, length, fs, win)
    n1 = normalize_signal(p1*theta + w1*(1 - theta), fs, win)
    return n1

# Log - Log Linear model
# Reference: https://tinyurl.com/loglogmodel
def curve_fit_log(xdata, ydata) :
    f = lambda x, b, m: m * x + b 
    
    xdata_log = np.log10(xdata) # Take Log of the x-coordinates
    ydata_log = np.log10(ydata) # Take Log of the y-coordinates
    
    # Fit linear model to log data
    popt_log, pcov_log = curve_fit(f, xdata_log, ydata_log) 

    # Generate the optimal line fitted to the model
    ydatafit_log = np.power(10, f(xdata_log, *popt_log))

    # Return the optimal parameters & fitted data (cov is not needed)
    return (popt_log, pcov_log, ydatafit_log)
