import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

import pystorm
from pystorm.hal.calibrator import Calibrator, PoolSpec

import utils

TAU_READOUT = 0.1

DATA_DIR = "data/test_accumulator_decodes/"
FIG_DIR = "figures/test_accumulator_decodes/"
DATA_FNAME = DATA_DIR + "snr_plot_(32, 32)_(0, 0).pck"
DATA_SNR_POI_FNAME = DATA_DIR + "snr_plot_(32, 32)_(0, 0)_poi_pred.pck"

###################################################################################################
# colors = mpl.cm.get_cmap('Purples')(np.linspace(0.5, 1, 3))[::-1]
colors = mpl.cm.get_cmap('Reds')(np.linspace(0.3, 0.7, 3))[::-1]

plt.style.use('/Users/samfok/Code/accumulator_decode/figures/ieee_tran.mplstyle')
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('text', usetex=True)
###################################################################################################

def plot_decode_stats(tau):
    # plot on gamma plot
    with open(DATA_FNAME, 'rb') as fhandle:
        data = pickle.load(fhandle)
    fend = data[0]
    fout_plot = data[1]
    fout_plt = data[2]
    snr_per = data[3]
    snr_poi = data[4]
    snr_gamma_min_plt = data[5]
    snr_gamma_max_plt = data[6]
    fmax_outs = data[7]
    fcn_fs = data[8]
    fouts = data[9]
    snrs = data[10]
    snr_pred_pers = data[11]
    xlim, ylim = data[-1]

    with open(DATA_SNR_POI_FNAME, 'rb') as fhandle:
        snr_pred_pois = pickle.load(fhandle)
    
    fig_snr, ax_snr = plt.subplots(figsize=(3.5, 3))
    ax_snr.grid(which="both", alpha=0.5)

    ax_snr.loglog(fout_plot*tau, snr_per, "k", linewidth=1)
    ax_snr.loglog(fout_plot*tau, snr_poi, "k", linewidth=1)
    fill_color = plt.get_cmap("Greys")(.1)
    ax_snr.fill_between(fout_plot*tau, snr_per, snr_poi, facecolor=fill_color)
    
    ax_snr.fill_between(
        fout_plt*tau, snr_gamma_max_plt, snr_gamma_min_plt,
        facecolor='#ff7f0e', alpha=0.5)
    ax_snr.loglog(fout_plt*tau, snr_gamma_max_plt, '-k', alpha=0.5)
    ax_snr.loglog(fout_plt*tau, snr_gamma_min_plt, '-k', alpha=0.5)
    
    len_c = len(colors)
    for idx0, fmax_out in enumerate(fmax_outs):
        for idx1, fcn_f in enumerate(fcn_fs):
            fout = fouts[idx0, idx1]
            snr = snrs[idx0, idx1]
            snr_pred_per = snr_pred_pers[idx0, idx1]
            snr_pred_poi = snr_pred_pois[idx0, idx1]

            if idx0 == 0:
                fcn_f_str = "".format(fcn_f)
                label_o = r"$f={:.0f}$".format(fcn_f)
            else:
                label_o = None
            # ax_snr.loglog(fout*tau, snr_pred_per, '_', color=colors[idx1%len_c], fillstyle='none')
            # ax_snr.loglog(fout*tau, snr_pred_poi, '_', color=colors[idx1%len_c], fillstyle='none')
            ax_snr.loglog([fout*tau, fout*tau], [snr_pred_poi, snr_pred_per], '_', color=colors[idx1%len_c], fillstyle='none')
            ax_snr.loglog([fout*tau, fout*tau], [snr_pred_poi, snr_pred_per], '-', color=colors[idx1%len_c], fillstyle='none')
            ax_snr.loglog(fout*tau, snr, 'o', color=colors[idx1%len_c], label=label_o)
    lgd = ax_snr.legend(bbox_to_anchor=(0.90, 0.05), loc="lower right")
    
    ax_snr.set_xlim((xlim[0], 400))
    ax_snr.set_ylim(ylim)
    ax_snr.set_xlabel(r"$F_\textrm{out}\cdot\tau$")
    ax_snr.set_ylabel("SNR")

    ax_snr.text(12.0, 3.95, "POISSON", rotation=20, rotation_mode='anchor')
    ax_snr.text(12, 46, "PERIODIC", rotation=35, rotation_mode='anchor')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig_snr.savefig(FIG_DIR + "snr_(32, 32)_(0, 0).pdf",
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
    
plot_decode_stats(TAU_READOUT)
# plt.show()
