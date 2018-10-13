import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

import pystorm
from pystorm.hal.calibrator import Calibrator, PoolSpec
from pystorm.hal.data_utils import lpf, bin_to_spk_times, bins_to_rates

import utils

# swept parameters

X = 16
Y = 16
# FMAX_OUTS = [500]
FMAX_OUTS = [1500, 1000, 500]

# X = 32
# Y = 32
# FMAX_OUTS = [1500, 1000, 500]
# # FMAX_OUTS = [2000, 1500, 1000]

# REG_L1_L2 = [0.1, 0.001, 0.00001, 1e-7, 1e-9]
REG_L1_L2 = [0.001,]

FCN_F = [1, 2, 4]
# FCN_F = [2.0]

# fixed parameters
# network parameters
NNEURON = X*Y
DIM = 1

DATA_DIR = "data/test_accumulator_decodes/"
FIG_DIR = "figures/test_accumulator_decodes/"

# experimental parameters
TRAINING_DOWNSTREAM_NS = 10000 # minimum time resolution, downstream time resolution
TRAINING_UPSTREAM_NS   = 1000000 # default upstream time resolution
FMAX_IN = 2000
TUNING_INPUT_POINTS = 240 # number of input points to take for collecting tuning data
TUNING_POINT_TIME_NS = int(0.5*1E9) # time to record data per tuning curve input point
VALIDATION_SKIP = 7 # reserve every VALIDATION_SKIP points from tuning data for validation
TEST_INPUT_POINTS = 121 # number of input points to take for validating fit
TEST_POINT_TIME_NS = int(0.5*1E9) # time to collect data for each testing point
TESTING_DOWNSTREAM_NS = 1000 # downstream time resolution for testing
TESTING_UPSTREAM_NS = 1000 # upstream time resolution for testing
TESTING_RAW_SPIKES_DOWNSTREAM_NS = 1000 # downstream time resolution for collecting raw spikes for testing
TESTING_RAW_SPIKES_UPSTREAM_NS = 10000 # downstream time resolution for collecting raw spikes for testing
CLIP_TIME = 0.2 # how much of the initial data to discard during testing
TAU_READOUT = 0.01 # synaptic readout time constant
FSCALE = 1000.

###################################################################################################

PS_ORIG = PoolSpec(
    label = "pool",
    YX = (Y, X),
    loc_yx = (0, 0),
    D = DIM,
)

def FCN(x, f):
    """Base function used in testing"""
    return 0.5 + 0.5*np.sin(np.pi*f*x)

class ExpData:
    def __init__(self, fmax_in, fmax_outs, fcn_fs):
        self.fmax_in = fmax_in
        self.fmax_outs = fmax_outs
        self.fcn_fs = fcn_fs

###################################################################################################

def get_nrmse(fit, target, normal_factor):
    assert fit.shape == target.shape
    return np.sqrt(np.mean((fit - target)**2))/normal_factor

###################################################################################################

EXP_DATA = ExpData(FMAX_IN, FMAX_OUTS, FCN_F)

###################################################################################################
def load_tuning_data(exp_data, fnames, valid_skip):
    """Load tuning data from a set of files"""
    if isinstance(fnames, str):
        fnames = [fnames]
    datasets = []
    for fname in fnames:
        with open(fname, 'rb') as fhandle:
            datasets += [pickle.load(fhandle)]
    input_rates = np.array([dataset.input_rates for dataset in datasets]).flatten().reshape((-1, 1))
    spike_rates = np.vstack(np.array([dataset.spike_rates for dataset in datasets]))
    n_input_points = len(input_rates)
    start_idx = valid_skip//2
    t_idx = np.ones(n_input_points, dtype=bool)
    t_idx[start_idx::valid_skip] = False
    v_idx = ~t_idx
    
    sort_idx = np.argsort(input_rates, axis=0).flatten()
    input_rates = input_rates[sort_idx]
    spike_rates = spike_rates[sort_idx]
    
    exp_data.training_input_rates = input_rates[t_idx]
    exp_data.training_spike_rates = spike_rates[t_idx]
    exp_data.validation_input_rates = input_rates[v_idx]
    exp_data.validation_spike_rates = spike_rates[v_idx]

###################################################################################################

class TuningData:
    def __init__(self, input_rates, spike_rates):
        self.input_rates = input_rates
        self.spike_rates = spike_rates
DATA_TUNING_DIR = "data/test_accumulator_decodes/tuning/"
DATA_FNAME_BASE = (DATA_TUNING_DIR + str(PS_ORIG.YX) + "_" + str(PS_ORIG.loc_yx) +
                   "_fmax_in_" + str(FMAX_IN) + "_samples_1200_sample_time_500ms_discard_0.5")
DATA_FNAMES = [
    DATA_FNAME_BASE + "_set_0.pck",
    DATA_FNAME_BASE + "_set_1.pck",
]
load_tuning_data(EXP_DATA, DATA_FNAMES, VALIDATION_SKIP)
###################################################################################################

def sweep_inputs_collect_decodes(
        input_points, fmax_in, time_per_point_ns, fname_pre_cache, fname_post_cache=""):
    """Collect decode data to test decoders"""
    time_points_ns = np.arange(input_points+1)*time_per_point_ns
    input_rates = np.zeros((input_points+1, 1))
    input_rates[:input_points, 0] = fmax_in * np.linspace(-1, 1, input_points)
    input_rates[-1, 0] = input_rates[-2, 0]

    fname_cache = (
        DATA_DIR + fname_pre_cache + "_" + str(PS_ORIG.YX) + "_" +
        str(PS_ORIG.loc_yx) + "_" + str(input_points) + "_" + str(time_per_point_ns) + fname_post_cache + ".pck")
    def sweep_fun():
        assert False
    output_rates = utils.cache_fun(fname_cache, sweep_fun)
    return input_rates[:-1], output_rates
###################################################################################################

def run_fits(exp_data):
    exp_data.decoders = []
    exp_data.training_targets = []
    exp_data.reg = np.zeros((len(exp_data.fmax_outs), len(exp_data.fcn_fs)))
    v_input = exp_data.validation_input_rates/exp_data.fmax_in
    v_spike_rates = exp_data.validation_spike_rates

    t_input = exp_data.training_input_rates/exp_data.fmax_in
    t_spike_rates = exp_data.training_spike_rates
    
    nrows = 2*len(exp_data.fmax_outs)
    ncols = len(exp_data.fcn_fs)
    for idx0, fmax_out in enumerate(exp_data.fmax_outs):
        exp_data.training_targets += [[]]
        exp_data.decoders += [[]]
        for idx1, fcn_f in enumerate(exp_data.fcn_fs):
            t_target = fmax_out * FCN(
                t_input, fcn_f) 
            print("Solving for fmax_out {:.0f} fcn_f {:.1f}".format(fmax_out, fcn_f))
            v_nrmse = np.zeros(len(REG_L1_L2))
            v_target = fmax_out * FCN(v_input, fcn_f)
            decoders_sweep = []
            for idx, reg in enumerate(REG_L1_L2):
                fname_cache = (
                    DATA_DIR + "decoders_" +
                    str(PS_ORIG.YX) + "_" + str(PS_ORIG.loc_yx) +
                    "_fmax_out_" + str(fmax_out) + "_fcn_f_" + str(fcn_f) +
                    "_L1L2_" + str(reg) + ".pck")
                def cached_fit_decoders():
                    assert False
                decoders, info = utils.cache_fun(fname_cache, cached_fit_decoders)
                t_fit = np.dot(t_spike_rates, decoders)
                v_fit = np.dot(v_spike_rates, decoders)
                v_nrmse[idx] = get_nrmse(v_fit, v_target, fmax_out)
                print("\treg {:.1e} v_nrmse {:.6f}".format(reg, v_nrmse[idx]))
                decoders_sweep.append(decoders)
            reg_idx = np.argmin(v_nrmse)
            exp_data.reg[idx0, idx1] = REG_L1_L2[reg_idx]
            exp_data.decoders[-1] += [decoders_sweep[reg_idx]]
            exp_data.training_targets[-1] += [t_target]
    exp_data.training_input = t_input
run_fits(EXP_DATA)

###################################################################################################

def test_decoders(exp_data):
    exp_data.test_input_rates = None
    exp_data.test_output_rates = []
    for idx0, fmax_out in enumerate(exp_data.fmax_outs):
        exp_data.test_output_rates += [[]]
        for idx1, fcn_f in enumerate(exp_data.fcn_fs):
            print("testing decoders optimized for fcn_f {:.1f} and fmax_out {:d}".format(fcn_f, fmax_out))
            test_input_rates, test_output_rates = sweep_inputs_collect_decodes(
                TEST_INPUT_POINTS,
                exp_data.fmax_in, TEST_POINT_TIME_NS, "testing",
                fname_post_cache="_fmax_in_{:d}_fcn_f_{:.1f}_fmax_out_{:d}_L1_{:f}_L2_{:f}".format(
                    exp_data.fmax_in, fcn_f, fmax_out, REG_L1_L2[0], REG_L1_L2[0],))
            if idx0 == 0 and idx1 == 0:
                exp_data.test_input_rates = test_input_rates
            exp_data.test_output_rates[idx0] += [test_output_rates]
test_decoders(EXP_DATA)

###################################################################################################

def check_plot_fits(exp_data):
    t_spike_rates = exp_data.training_spike_rates
    v_spike_rates = exp_data.validation_spike_rates
    
    t_input = exp_data.training_input_rates/exp_data.fmax_in
    v_input = exp_data.validation_input_rates/exp_data.fmax_in
    tst_input = exp_data.test_input_rates/exp_data.fmax_in
    
    exp_data.t_nrmse = np.zeros((len(exp_data.fmax_outs), len(exp_data.fcn_fs)))
    exp_data.v_nrmse = np.zeros_like(exp_data.t_nrmse)
    exp_data.vq_nrmse = np.zeros_like(exp_data.t_nrmse)
    exp_data.tst_nrmse = np.zeros_like(exp_data.t_nrmse)
    
    stim0_idx = np.argmin(np.abs(exp_data.training_input_rates)) # omit neurons that don't fire at 0
    spike_rates0 = exp_data.training_spike_rates[stim0_idx]
    nz_nrn0_idx = spike_rates0 > 0.
    
    hist_bins = np.linspace(-1, 1, 50)
    
    q_levels = 64
    q_res = 1/q_levels
    q_bins = np.hstack((-np.arange(q_levels+1)[::-1], np.arange(q_levels+1)[1:]))*q_res
    nrows = 4*len(exp_data.fmax_outs)
    ncols = len(exp_data.fcn_fs)
    exp_data.tst_target = []
    exp_data.tst_observed = []
    for idx0, fmax_out in enumerate(exp_data.fmax_outs):
        exp_data.tst_target += [[]]
        exp_data.tst_observed += [[]]
        for idx1, fcn_f in enumerate(exp_data.fcn_fs):
            decoders = exp_data.decoders[idx0][idx1]
            q_idx = np.digitize(decoders-q_res/2, q_bins)
            q_decoders = q_bins[q_idx].reshape(decoders.shape)
            nz_decoders = decoders[np.logical_and(np.abs(decoders) >= 1/8192, nz_nrn0_idx.reshape(-1, 1))]
            
            t_target = exp_data.training_targets[idx0][idx1]
            v_target = fmax_out * FCN(v_input, fcn_f)
            tst_target = fmax_out * FCN(tst_input, fcn_f)
            
            t_fit = np.dot(t_spike_rates, decoders)
            v_fit = np.dot(v_spike_rates, decoders)
            vq_fit = np.dot(v_spike_rates, q_decoders)
            tst_observed = exp_data.test_output_rates[idx0][idx1]
            
            t_nrmse = get_nrmse(t_fit, t_target, fmax_out)
            v_nrmse = get_nrmse(v_fit, v_target, fmax_out)
            vq_nrmse = get_nrmse(vq_fit, v_target, fmax_out)
            tst_nrmse = get_nrmse(tst_observed, tst_target, fmax_out)
            
            exp_data.t_nrmse[idx0, idx1] = t_nrmse
            exp_data.v_nrmse[idx0, idx1] = v_nrmse
            exp_data.vq_nrmse[idx0, idx1] = vq_nrmse
            exp_data.tst_nrmse[idx0, idx1] = tst_nrmse
            
            exp_data.tst_target[-1] += [tst_target]
            exp_data.tst_observed[-1] += [tst_observed]
            
check_plot_fits(EXP_DATA)

###################################################################################################
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[0], colors[2], colors[1]]

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

def plot_figure(exp_data):
    tst_input = exp_data.test_input_rates/exp_data.fmax_in
    
    stim0_idx = np.argmin(np.abs(exp_data.training_input_rates)) # omit neurons that don't fire at 0
    spike_rates0 = exp_data.training_spike_rates[stim0_idx]
    nz_nrn0_idx = spike_rates0 > 0.
    
    hist_bins = np.linspace(-1, 1, 40)
    
    nrows = 2
    ncols = len(exp_data.fcn_fs)
    # fig, axs = plt.subplots(nrows, ncols, figsize=(3.5, 2))
    # fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*3+1,nrows*2+1))
    fig = plt.figure(figsize=(3.5, 1.75))
    gs = mpl.gridspec.GridSpec(2, 3, height_ratios=[2, 1])
    axs = [[plt.subplot(gs[idx0, idx1]) for idx1 in range(ncols)] for idx0 in range(nrows)]
    if ncols==1:
        axs = np.array([axs]).T.tolist()
    for idx1, fcn_f in enumerate(exp_data.fcn_fs):
        tst_target = FCN(tst_input, fcn_f)
        axs[0][idx1].plot(tst_input, tst_target, 'k', linewidth=0.5, alpha=0.5)
    for idx0, fmax_out in enumerate(exp_data.fmax_outs):
        for idx1, fcn_f in enumerate(exp_data.fcn_fs):
            decoders = exp_data.decoders[idx0][idx1]
            nz_decoders = decoders[np.logical_and(np.abs(decoders) >= 1/64, nz_nrn0_idx.reshape(-1, 1))]
            axs[0][idx1].plot(tst_input, exp_data.test_output_rates[idx0][idx1]/fmax_out,
                              color=colors[idx0], lw=1,
                              label="{:d}".format(fmax_out))
            axs[1][idx1].hist(nz_decoders, bins=hist_bins,
                              color=colors[idx0],
                              alpha=0.5, label="{:d}".format(fmax_out))
    def set_axs_fmt(fig, axs, exp_data):
        ylim_mins = np.min(
            np.array([[axs[idx0][idx1].get_ylim()[0] for idx1 in range(len(exp_data.fcn_fs))]
                     for idx0 in range(2)]), axis=1)
        ylim_maxes = np.max(
            np.array([[axs[idx0][idx1].get_ylim()[1] for idx1 in range(len(exp_data.fcn_fs))]
                     for idx0 in range(2)]), axis=1)
        axs[0][0].tick_params(direction='in')
        axs[1][0].tick_params(direction='in')
        for idx0 in range(2):
            for idx1 in range(len(exp_data.fcn_fs))[1:]:
                axs[idx0][idx1].set_ylim(ylim_mins[idx0], ylim_maxes[idx0])
        for idx1 in range(len(exp_data.fcn_fs)):
            axs[0][idx1].set_yticks([0, 0.5, 1])
        for idx1 in range(len(exp_data.fcn_fs))[1:]:
            axs[0][idx1].set_yticklabels([])
            axs[1][idx1].set_yticklabels([])
        for idx1 in range(len(exp_data.fcn_fs)):
            axs[0][idx1].set_xticklabels([r"$-\!1$", "0", "1"])
            axs[1][idx1].set_xticklabels([r"$-\!1$", "0", "1"])
        for idx1, fcn_f in enumerate(exp_data.fcn_fs):
            ax_pos = axs[0][idx1].get_position()
            colors = [axs[0][idx1].get_lines()[idx0].get_color()
                      for idx0 in range(1, len(exp_data.fmax_outs)+1)]
            ymax = axs[0][idx1].get_ylim()[1]
            if len(exp_data.fmax_outs) == 3:
                axs[0][idx1].text(-1, 1.05*ymax, "{:.3f}".format(
                    exp_data.tst_nrmse[0][idx1]), ha="left", va="bottom", color=colors[0])
                axs[0][idx1].text(0, 1.05*ymax, "{:.3f}".format(
                    exp_data.tst_nrmse[1][idx1]), ha="center", va="bottom", color=colors[1])
                axs[0][idx1].text(1, 1.05*ymax, "{:.3f}".format(
                    exp_data.tst_nrmse[2][idx1]), ha="right", va="bottom", color=colors[2])
            axs[0][idx1].set_xlabel(r"$x$", labelpad=-1.1)
            axs[1][idx1].set_xlabel(r"$w$", labelpad=-1.1)
            axs[0][idx1].tick_params(direction='in', pad=1.5)
            axs[1][idx1].tick_params(direction='in', pad=1.5)
            axs[0][idx1].set_xlim([-1, 1])
            axs[1][idx1].set_xlim([-1, 1])
        # axs[0][0].legend(loc="upper left", title=r"$F_\textrm{out}$")
        axs[0][0].set_ylabel(r"$y_f/F_\textrm{out}$")
        axs[1][0].set_ylabel("Count")
    set_axs_fmt(fig, axs, exp_data)
    plt.subplots_adjust(left=-0.05, bottom=-0.08, right=1.05, top=1.08, wspace=0.10, hspace=0.35)
    fig.text(-0.165, 1.166, "RMSE:", fontsize=SMALL_SIZE, ha="left", va="top")
    # for fmt_str in ['.png']:
    for fmt_str in ['.png', '.pdf']:
        print("saving {} figure".format(fmt_str))
        fig.savefig(FIG_DIR + "test_decode_weights_" + str(PS_ORIG.YX) + "_" + str(PS_ORIG.loc_yx) +
                    "_fcn_f_" + str(FCN_F) + "_fmax_out_" + str(FMAX_OUTS) + fmt_str,
                   bbox_inches="tight", transparent=True)
    # print("saving {} figure".format(".svg"))
    # fig.savefig(FIG_DIR + "test_decode_weights_" + str(PS_ORIG.YX) + "_" + str(PS_ORIG.loc_yx) +
    #             "_fcn_f_" + str(FCN_F) + "_fmax_out_" + str(FMAX_OUTS) + ".svg",
    #            format="svg", bbox_inches="tight")
            

plot_figure(EXP_DATA)
# print("showing figure")
# plt.show()
###################################################################################################
