__all__ = ['Compose', 'RandomCrop', 'CenterCrop', 'FixedCrop', 'GaussianNoise', 'Resample', 'ToSpectrogram', 'Flatten', 'ToTensor', 'Normalize', 'NormalizeBatch', 'ButterFilter', 'ChannelFilter', 'Transform', 'StaticTransform', 'TupleTransform', 'SequenceToSampleLabelTransform', 'RandomLeadMasking', 'FixedLeadMasking', 'parse_lead_spec', 'CANONICAL_LEAD_NAMES']
# PROPOSED: add 'ExtractSingleLead' to __all__ once the class above is uncommented
__all__.append('ExtractSingleLead')
__all__.append('SingleLeadToSlot')


import numpy as np
import torch
import torch.utils.data

import math
import random
import resampy

#from skimage import transform

#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

from scipy.signal import butter, sosfilt, sosfiltfilt
from scipy import signal

#from scipy.interpolate import interp1d



#def nn_upsample(xin, yin, xout):
#    '''performs nearest neighbor upsampling of the integer array yin with values at xin for new datapoints at xout'''
#    f = interp1d(xin,yin, kind="nearest",bounds_error=False,fill_value="extrapolate")
#    return f(xout).astype(np.int64)

#def resample_labels(startpts, labels, startpts_to_mid, startpts_new, startpts_to_mid_new):
#    '''resamples integer labels labels at starpts+startpts_to_mid to new anchor points at startpts_new+startpts_to_mid_new'''
#    if(isinstance(startpts_to_mid,float) or isinstance(startpts_to_mid,int)):
#        startpts_to_mid = np.ones_like(startpts)*startpts_to_mid
#    if(isinstance(startpts_to_mid_new,float) or isinstance(startpts_to_mid_new,int)):
#        startpts_to_mid_new = np.ones_like(startpts_new)*startpts_to_mid_new
#    midpts = np.array(startpts)+startpts_to_mid
#    midpts_new = np.array(startpts_new)+startpts_to_mid_new
#    return nn_upsample(midpts, labels, midpts_new)

#https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_filter(lowcut=10, highcut=20, fs=50, order=5, btype='band'):
    '''returns butterworth filter with given specifications'''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos = butter(order, [low, high] if btype=="band" else (low if btype=="low" else high), analog=False, btype=btype, output='sos')
    return sos

#def butter_filter_frequency_response(filter):
#    '''returns frequency response of a given filter (result of call of butter_filter)'''
#    w, h = sosfreqz(filter)
#    #gain vs. freq(Hz)
#    #plt.plot((fs * 0.5 / np.pi) * w, abs(h))
#    return w,h
#
#def apply_butter_filter(data, filter, forwardbackward=True):
#    '''pass filter from call of butter_filter to data (assuming time axis at dimension 0)'''
#    if(forwardbackward):
#        return sosfiltfilt(filter, data, axis=0)
#    else:
#        data = sosfilt(filter, data, axis=0)

class Compose:
    '''composes several transformations into a single one (as provided by torchvision.transforms.Compose)'''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inp):
        for t in self.transforms:
            inp = t(inp)
        return inp


class RandomCrop(object):
    """Crop randomly from a sample.
    """

    def __init__(self, output_size,annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample

        timesteps= len(data)
        assert(timesteps>=self.output_size)
        if(timesteps==self.output_size):
            start=0
        else:
            start = random.randint(0, timesteps - self.output_size-1) #np.random.randint(0, timesteps - self.output_size)
            idxs[1]+=start
            idxs[2]=idxs[1]+self.output_size
        
        data = data[start: start + self.output_size]
        if(self.annotation):
            label = label[start: start + self.output_size]

        return (data, label, static, static_cat, idxs)


class CenterCrop(object):
    """Center crop from a sample.
    """

    def __init__(self, output_size, annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample

        timesteps= len(data)
        start = (timesteps - self.output_size)//2
        idxs[1] += start
        idxs[2] = idxs[1] + self.output_size

        data = data[start: start + self.output_size]
        if(self.annotation):
            label = label[start: start + self.output_size]

        return (data, label, static, static_cat, idxs)

class FixedCrop(object):
    """Take a fixed crop from a sample (for aligned data e.g. spectrograms). start_idx and end_idx are relative to the start of the respective sample
    """

    def __init__(self, start_idx, end_idx, annotation=False):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.output_size = self.end_idx-self.start_idx
        self.annotation = annotation

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        assert(self.end_idx<len(data))
        start = self.start_idx
        idxs[1] += self.start_idx
        idxs[2] = idxs[1] + self.output_size

        data = data[start: start + self.output_size]
        if(self.annotation):
            label = label[start: start + self.output_size]

        return (data, label, static, static_cat, idxs)

class GaussianNoise(object):
    """Add gaussian noise to sample.
    """

    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, sample):
        if self.scale ==0:
            return sample
        else:
            data, label, static, static_cat, idxs = sample
            data = data + np.reshape(np.array([random.gauss(0,self.scale) for _ in range(np.prod(data.shape))]),data.shape)#np.random.normal(scale=self.scale,size=data.shape).astype(np.float32)
            return (data, label, static, static_cat, idxs)


#class Rescale(object):
#    """Resample by factor.
#    """
#
#    def __init__(self, scale=0.5,interpolation_order=3):
#        self.scale = scale
#        self.interpolation_order = interpolation_order
#
#    def __call__(self, sample):
#        if self.scale ==1:
#            return sample
#        else:
#            data, label, static, static_cat, idxs = sample
#            timesteps_new = int(self.scale * len(data))
#            data = transform.resize(data,(timesteps_new,data.shape[1]),order=self.interpolation_order).astype(np.float32)
#            return (data, label, static, static_cat, idxs)

class Resample(object):
    """Resample on the fly
    """

    def __init__(self, fs_source, fs_target):
        self.fs_source = fs_source
        self.fs_target = fs_target

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        data = resampy.resample(data, self.fs_source, self.fs_target, axis=0)
        return (data, label, static, static_cat, idxs)

class ToSpectrogram(object):
    """Convert to spectrogram on the fly
    """

    def __init__(self, fs, window="hann", nperseg=256, noverlap= 128, pad="reflect", log_transform=True):
        self.fs = fs # Sampling frequency
        self.window = window # Window function
        self.nperseg = nperseg # Length of each segment
        self.noverlap = noverlap # Number of points to overlap between segments
        self.pad = pad #
        self.log_transform = log_transform
        self.eps = np.finfo(float).eps  # get the smallest representable float value

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        #data has shape ts, ch
        
        #was: window_size=2, overlap=1
        #signals = np.pad(data, ((int(0.5*self.fs), int(0.5*self.fs)), (0, 0)))
        #_, _, Zxx = signal.spectrogram(signals.T, fs=self.fs, window=signal.windows.hamming(int(self.window_size * self.fs)), noverlap=int(self.fs*self.overlap), nfft=self.nfft)
        
        # Pad the signal
        if(self.pad is not None):
            pad_length = self.nperseg // 2
            data = np.pad(data, ((pad_length, pad_length), (0, 0)), mode='reflect')

        # Calculate the spectrogram for all channels at once
        _, _, Sxx = signal.spectrogram(data, fs=self.fs, window=self.window, 
                               nperseg=self.nperseg, noverlap=self.noverlap, axis=0)
        
        #originally Sxx has shape [freq, ch, ts]
        Sxx = np.transpose(Sxx, (2,1,0))
        if(self.log_transform):
            Sxx = 20 * np.log10(np.abs(Sxx) + self.eps)
    
        #print("spec",Sxx.shape)
        return (Sxx, label, static, static_cat, idxs)

class Flatten(object):
    """flatten all channel dimensions e.g. to process spectrograms with 1d convs
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        #data has shape ts, ch1, ch2,...
        data = data.reshape(data.shape[0], -1)
        return (data, label, static, static_cat, idxs)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, transpose_data=True, transpose_label=False):
        #swap channel and time axis for direct application of pytorch's convs
        self.transpose_data=transpose_data
        self.transpose_label=transpose_label

    def __call__(self, sample):

        def _to_tensor(data,transpose=False):
            if(isinstance(data,np.ndarray) or isinstance(data,list)):
                if(transpose):#seq,[x,y,]ch -> [x,y,]ch, seq
                    return torch.from_numpy(np.moveaxis(data,0,-1))
                else:
                    return torch.from_numpy(data)
            else:#default_collate will take care of it
                return data

        data, label, static, static_cat, idxs = sample
        if not isinstance(data,tuple):
            data = _to_tensor(data,self.transpose_data)
        else:
            data = tuple(_to_tensor(x,self.transpose_data) for x in data)

        if not isinstance(label,tuple):
            label = _to_tensor(label,self.transpose_label)
        else:
            label = tuple(_to_tensor(x,self.transpose_label) for x in label)

        if not isinstance(static,tuple):
            static = _to_tensor(static)
        else:
            static = tuple(_to_tensor(x) for x in static)

        if not isinstance(static_cat,tuple):
            static_cat = _to_tensor(static_cat)
        else:
            static_cat = tuple(_to_tensor(x) for x in static_cat)
        
        if not isinstance(idxs,tuple):
            idxs = _to_tensor(idxs)
        else:
            idxs = tuple(_to_tensor(x) for x in idxs)
            

        return (data, label, static, static_cat, idxs) #returning as a tuple (potentially of lists)


class Normalize(object):
    """Normalize using given stats.
    """
    def __init__(self, stats_mean, stats_std, input=True, channels=[]):
        self.stats_mean=stats_mean.astype(np.float32) if stats_mean is not None else None
        self.stats_std=stats_std.astype(np.float32)+1e-8 if stats_std is not None else None
        self.input = input
        if(len(channels)>0):
            for i in range(len(stats_mean)):
                if(not(i in channels)):
                    self.stats_mean[:,i]=0
                    self.stats_std[:,i]=1

    def __call__(self, sample):
        datax, labelx, static, static_cat, idxs = sample
        data = datax if self.input else labelx
        #assuming channel last
        if(self.stats_mean is not None):
            data = data - self.stats_mean
        if(self.stats_std is not None):
            data = data/self.stats_std

        if(self.input):
            return (data, labelx, static, static_cat, idxs)
        else:
            return (datax, data, static, static_cat, idxs)


class NormalizeBatch(object):
    """Normalize using batch statistics.
    axis: tuple of integers of axis numbers to be normalized over (by default everything but the last)
    """
    def __init__(self, input=True, channels=[],axis=None):
        self.channels = channels
        self.channels_keep = None
        self.input = input
        self.axis = axis

    def __call__(self, sample):
        datax, labelx, static, static_cat, idxs = sample
        data = datax if self.input else labelx
        #assuming channel last
        #batch_mean = np.mean(data,axis=tuple(range(0,len(data)-1)))
        #batch_std = np.std(data,axis=tuple(range(0,len(data)-1)))+1e-8
        batch_mean = np.mean(data,axis=self.axis if self.axis is not None else tuple(range(0,len(data.shape)-1)))
        batch_std = np.std(data,axis=self.axis if self.axis is not None else tuple(range(0,len(data.shape)-1)))+1e-8

        if(len(self.channels)>0):
            if(self.channels_keep is None):
                self.channels_keep = np.setdiff(range(data.shape[-1]),self.channels)

            batch_mean[self.channels_keep]=0
            batch_std[self.channels_keep]=1

        data = (data - batch_mean)/batch_std

        if(self.input):
            return (data, labelx, static, static_cat, idxs)
        else:
            return (datax, data, static, static_cat, idxs)


class ButterFilter(object):
    """Apply filter
    """

    def __init__(self, lowcut=50, highcut=50, fs=100, order=5, btype='band', forwardbackward=True, input=True):
        self.filter = butter_filter(lowcut,highcut,fs,order,btype)
        self.input = input
        self.forwardbackward = forwardbackward

    def __call__(self, sample):
        datax, labelx, static, static_cat, idxs = sample
        data = datax if self.input else labelx

        if(self.forwardbackward):
            data = sosfiltfilt(self.filter, data, axis=0)
        else:
            data = sosfilt(self.filter, data, axis=0)

        if(self.input):
            return (data, labelx, static, static_cat, idxs)
        else:
            return (datax, data, static, static_cat, idxs)


class ChannelFilter(object):
    """Select certain channels.
    axis: axis index of the channel axis
    """

    def __init__(self, channels=[0], axis=-1, input=True):
        self.channels = channels
        self.input = input
        self.axis = axis

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        if(self.input):
            return (np.take(data,self.channels,axis=self.axis), label, static, static_cat, idxs) #(data[...,self.channels], label, static)
        else:
            return (data, np.take(label,self.channels,axis=self.axis), static, static_cat, idxs)


class Transform(object):
    """Transforms data using a given function i.e. data_new = func(data) for input is True else label_new = func(label)
    """

    def __init__(self, func, input=False):
        self.func = func
        self.input = input

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        if(self.input):
            return (self.func(data), label, static, static_cat, idxs)
        else:
            return (data, self.func(label), static, static_cat, idxs)

class StaticTransform(object):
    """Transforms static data using a given function i.e. data_new = func(data) for input is True else label_new = func(label)
    """
    def __init__(self, func):
        self.func = func
        
    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        static, static_cat = self.func(static, static_cat)
        return (data, label, static, static_cat, idxs)

class TupleTransform(object):
    """Transforms data using a given function (operating on both data and label and return a tuple) i.e. data_new, label_new = func(data_old, label_old)
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, sample):
        data,label,static, static_cat,idxs = sample
        data2, label2 = self.func(data,label,static,static_cat)
        return  (data2, label2, static, static_cat, idxs)

class SequenceToSampleLabelTransform(object):
    """Transforms sequence-level to sample-level labels
    majority vote: pick the most frequent label as segment label (i.e. suitable for single-label classification)
    num_classes: number of output classes
    binary: binary instead of floating point outputs (where the latter represent fractions)
    epoch_length: split the original sequence in ts//epoch_length fragments
    """

    def __init__(self, majority_vote=False, num_classes=2, binary=False,epoch_length=0):
        self.majority_vote = majority_vote
        self.num_classes = num_classes
        self.binary = binary
        self.epoch_length = epoch_length

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        
        epoch_length = self.epoch_length if self.epoch_length>0 else len(label)
        if(len(label.shape)==1):#each time step is single-labeled
            label = np.eye(self.num_classes)[label] #now label has shape ts,num_classes
        cnts = np.sum(label.reshape((-1,epoch_length,label.shape[-1])),axis=1)#segments,classes
        if(self.majority_vote):
            label = np.argmax(cnts,axis=-1)#segments
        else:
            if(self.binary):
                label=(cnts>0).astype(np.float32)
            else:
                label = (cnts/epoch_length).astype(np.float32)
        if(self.epoch_length>0):
            return (data, label, static, static_cat, idxs)
        else:#just one segment
            return (data, label[0], static, static_cat, idxs)

#: Canonical lead order of the preprocessed memmaps (see
#: preprocess_ecg_dataset.ipynb: channel_stoi_canonical). Index 0 is lead I,
#: index 1 is lead II, etc.
CANONICAL_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
                        "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_lead_spec(spec, num_leads=12):
    """Parse a lead specification into a sorted list of 0-based lead indices.

    Accepts a comma-separated string (or an iterable) mixing canonical lead
    names and 0-based indices, case-insensitively:
        "I"        -> [0]
        "II"       -> [1]
        "I,II"     -> [0, 1]
        "0,1"      -> [0, 1]
        "i, v5"    -> [0, 10]
    """
    if isinstance(spec, str):
        items = [s.strip() for s in spec.split(",")]
    else:
        items = [str(s).strip() for s in spec]

    lookup = {n.lower(): i for i, n in enumerate(CANONICAL_LEAD_NAMES)}
    leads = []
    for item in items:
        if item == "":
            continue
        if item.lower() in lookup:
            idx = lookup[item.lower()]
        else:
            try:
                idx = int(item)
            except ValueError:
                raise ValueError(
                    f"Unknown lead '{item}'. Use a 0-based index or one of "
                    f"{CANONICAL_LEAD_NAMES}."
                )
        if idx < 0 or idx >= num_leads:
            raise ValueError(f"Lead index {idx} out of range [0,{num_leads - 1}]")
        leads.append(idx)

    if not leads:
        raise ValueError(f"Empty lead specification: {spec!r}")
    return sorted(set(leads))


class FixedLeadMasking(object):
    """Keep an explicitly specified set of leads and zero out all the others.

    Deterministic counterpart of RandomLeadMasking: instead of sampling k leads
    per sample, the *same* leads are kept for every sample, which is what
    "evaluate on lead I only / lead II only" requires (no seeds needed).

    Zeroing (rather than slicing) keeps the 12-channel input interface the
    baselines expect; RELAY's wrapper detects the zeroed leads and builds a
    genuine k-lead graph, exactly as it does under RandomLeadMasking.

    Note: unlike RandomLeadMasking, no non-zero-lead fallback is applied — the
    requested lead is kept as recorded, so a recording whose requested lead is
    flat yields an all-zero input for every model alike.
    """

    def __init__(self, leads, axis=-1, expected_leads=12):

        self.leads = parse_lead_spec(leads, num_leads=expected_leads or 12)
        self.axis = axis
        self.expected_leads = expected_leads

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample

        axis = self.axis % data.ndim
        num_leads = data.shape[axis]

        if self.expected_leads is not None:
            assert num_leads == self.expected_leads, f"Expected {self.expected_leads} leads, but got {num_leads}"

        if max(self.leads) >= num_leads:
            raise ValueError(
                f"Requested leads {self.leads} but the recording has only {num_leads}"
            )

        if len(self.leads) == num_leads:
            return (data, label, static, static_cat, idxs)

        # Make a copy to avoid modifying memmaps or read-only arrays
        data = data.copy()

        lead_mask = np.zeros(num_leads, dtype=bool)
        lead_mask[self.leads] = True

        # Broadcast the per-lead mask across the full tensor and zero out
        # non-selected leads in one vectorised pass.
        mask_shape = [1] * data.ndim
        mask_shape[axis] = num_leads
        keep_mask = lead_mask.reshape(mask_shape)
        np.copyto(data, 0, where=~keep_mask)

        return (data, label, static, static_cat, idxs)


class SingleLeadToSlot(object):
    """Places a single-channel recording into one lead slot of a zero-filled multi-lead input.

    For single-lead datasets (e.g. CinC 2017, Icentia11k) evaluated with models expecting 12 leads:
    (T, 1) -> (T, num_leads), the recorded channel in column `slot` and zeros elsewhere. Models that
    detect zeroed leads (RELAY, ST-MEM with --st-mem-native-leads) then see a genuine 1-lead input,
    tagged with the slot's lead identity where the model uses one.
    """

    def __init__(self, slot, axis=-1, num_leads=12):
        leads = parse_lead_spec(slot, num_leads=num_leads)
        assert len(leads) == 1, f"single-lead slot must be one lead, got {slot}"
        self.slot = leads[0]
        self.axis = axis
        self.num_leads = num_leads

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        axis = self.axis % data.ndim
        assert data.shape[axis] == 1, f"SingleLeadToSlot expects 1 channel, got {data.shape[axis]}"
        shape = list(data.shape)
        shape[axis] = self.num_leads
        out = np.zeros(shape, dtype=data.dtype)
        index = [slice(None)] * data.ndim
        index[axis] = slice(self.slot, self.slot + 1)
        out[tuple(index)] = data
        return (out, label, static, static_cat, idxs)


class RandomLeadMasking(object):
    """Keeps k random leads and zeroes the others. lead_pool (optional list of lead indices) restricts the
    draw to those leads, e.g. the 8 independent leads for a model that discards III/aVR/aVL/aVF."""

    def __init__(self, k=None, axis=-1, expected_leads=12, lead_pool=None):
        
        self.k = k
        self.axis = axis
        self.expected_leads = expected_leads
        self.lead_pool = None if lead_pool is None else sorted(set(lead_pool))
        
    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample

        axis = self.axis % data.ndim
        num_leads = data.shape[axis]

        if self.expected_leads is not None:
            assert num_leads == self.expected_leads, f"Expected {self.expected_leads} leads, but got {num_leads}"

        k = self.k if self.k is not None else random.randint(1, num_leads)

        if k < 1 or k > num_leads:
            raise ValueError(f"k must be between 1 and {num_leads}, got {k}")

        if k == num_leads:
            return (data, label, static, static_cat, idxs)

        # Make a copy to avoid modifying memmaps or read-only arrays
        data = data.copy()

        # Sample only among leads that actually carry signal: some recordings
        # contain flat/disconnected channels stored as all-zero, and picking
        # those as kept leads would produce an entirely-zero sample.
        other_axes = tuple(i for i in range(data.ndim) if i != axis)
        nonzero_mask = np.any(data != 0, axis=other_axes) if other_axes else (data != 0)
        nonzero_leads = np.flatnonzero(nonzero_mask)
        if self.lead_pool is not None:
            nonzero_leads = np.array([l for l in nonzero_leads if l in self.lead_pool], dtype=int)

        # Clamp k: if the recording has fewer non-zero leads than requested,
        # keep all available non-zero leads (the rest are already zero).
        if len(nonzero_leads) == 0:
            return (data, label, static, static_cat, idxs)
        k = min(k, len(nonzero_leads))

        leads_to_keep = random.sample(nonzero_leads.tolist(), k)

        lead_mask = np.zeros(num_leads, dtype=bool)
        lead_mask[leads_to_keep] = True

        # Broadcast the per-lead mask across the full tensor and zero out
        # non-selected leads in one vectorised pass.
        mask_shape = [1] * data.ndim
        mask_shape[axis] = num_leads
        keep_mask = lead_mask.reshape(mask_shape)
        np.copyto(data, 0, where=~keep_mask)

        return (data, label, static, static_cat, idxs)


# --- PROPOSED ADDITION: ExtractSingleLead transform ---
# This transform extracts ONE lead from the multi-lead matrix, reducing the channel
# dimension from N to 1 (e.g. (T,12) → (T,1)).  Required when loading
# 1_lead_ECGFounder.pth because Net1D must be instantiated with in_channels=1.
#
# Two modes:
#   limb_leads_only=True  → sample from indices [0, 5] inclusive (I, II, III, aVR, aVL, aVF)
#                           use when 1_lead_ECGFounder.pth is loaded WITHOUT --random_lead_sampling
#   limb_leads_only=False → sample from all indices [0, num_leads-1]
#                           use when --random_lead_sampling is flagged and --k=1 with 1-lead model
#
class ExtractSingleLead(object):
    """Extract a single random lead; reduces the lead axis from N to 1."""
    def __init__(self, limb_leads_only=True, axis=-1):
        self.limb_leads_only = limb_leads_only
        self.axis = axis

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        num_leads = data.shape[self.axis]
        lead_pool = list(range(0, 6)) if self.limb_leads_only else list(range(0, num_leads))
        chosen = random.choice(lead_pool)
        if self.axis == -1:
            data = data[..., chosen:chosen+1]   # (T, 1)
        else:
            data = np.take(data, [chosen], axis=self.axis)  # (1, T)
        return (data, label, static, static_cat, idxs)
# --- END PROPOSED ADDITION ---
