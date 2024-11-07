import numpy as np
from scipy import signal
# import tensorflow as tf
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact 

def gen_continous_signal(t, time_shift=0.0, freq=1.0, amplitude=1.0, signal_type='sinusoidal'):
    switcher = {
        'sinusoidal': lambda: amplitude * np.sin(2 * np.pi * freq * (t - time_shift)),
        'sawtooth': lambda: amplitude * signal.sawtooth(2 * np.pi * freq * (t - time_shift)),
        'square': lambda: amplitude * signal.square(2 * np.pi * freq * (t - time_shift)),
        'gauss_pulse': lambda: amplitude * signal.gausspulse(t-time_shift, fc=freq)
    }
    # Gọi hàm lambda để lấy kết quả
    return switcher.get(signal_type, lambda: amplitude * np.sin(2 * np.pi * freq * (t - time_shift)))()

# 1. Lấy mẫu (Sampling)
def sampling(continuous_signal, sampling_rate, duration):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    return t, continuous_signal(t)

def min_max_norm(x, max_val=None, min_val=None):

    if (max_val == None) or (min_val == None):
        norm_x = np.float32(x - np.min(x))/(np.max(x)-np.min(x))
    else: 
        norm_x = np.float32(x - min_val)/(max_val - min_val)
    return norm_x

def z_score_norm(x, mean = 0, deviation = 1):
    return (x - mean)/deviation

def rms_norm(x):
    rms = np.sqrt(np.sum(x**2))/ len(x)
    return x / rms

def max_abs_norm(x, max_abs=None):
    if (max_abs == None) or (max_abs == 0.0): 
        norm_x = x/ np.max(np.abs(x))
    else:
        max_abs = np.abs(max_abs)
        norm_x = x / max_abs
    return norm_x


def normalizing_signal(signal, normalized_type = 'min_max'):
    switcher = {
        'min_max': lambda: min_max_norm(signal),
        'z_score': lambda: z_score_norm(signal),
        'rms': lambda: rms_norm(signal),
        'max_abs': lambda: max_abs_norm(signal)
    }

    return switcher.get(normalized_type, lambda: min_max_norm(signal))()


# 2. Lượng tử hóa (Quantization)
def quantization(signal, levels, min_range=0.0, max_range=1.0, normalized_type = 'min_max'):
    # print(f'levels = {levels}')
    sig_list = []
    
    if not (normalized_type == None):
        normalized_signal = normalizing_signal(signal, normalized_type)
        clipped_signal = np.clip(normalized_signal, min_range, max_range)
    else:
        clipped_signal = signal
        normalized_signal =signal
    
    quantized_signal = np.round(clipped_signal * np.float32(levels-1)) / np.float32(levels-1)
    sig_list.append(quantized_signal)
    sig_list.append(normalized_signal)
    sig_list.append(clipped_signal)
    sig_list.append(min_range)
    sig_list.append(max_range)
    sig_list.append(levels)
    return sig_list

def quantization_np(signal, levels, min_range=0, max_range=1, normalized_type = 'min_max'):
    normalized_signal = normalizing_signal(signal, normalized_type)
    bins = np.linspace(min_range, max_range, num=levels)
    quantized_signal = np.digitize(normalized_signal, bins, right=False)
    # quantized_signal = np.digitize(signal, bins, right=True)
    return quantized_signal, min_range, max_range

def signal_description(signal_type='sinusoidal', amplitude=1.0, freq= 1.0, time_shift=0.0):
    signal = dict()
    signal['signal_type'] = signal_type
    signal['amplitude'] = amplitude
    signal['frequency'] = freq
    signal['time_shift'] = time_shift
    return signal

def composite_signal(signal_list, duration=1.0, sampling_rate=50.0):
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    tot_sig = 0
    for ind, src in enumerate(signal_list):
        sig_type = src['signal_type']
        amp = src['amplitude']
        freq = src['frequency']
        t_sh = src['time_shift']
        
        # Thêm amplitude vào hàm gen_continous_signal
        cont_sig = gen_continous_signal(t, time_shift=t_sh, freq=freq, amplitude=amp, signal_type=sig_type)
        tot_sig += cont_sig
    return tot_sig, t

def alias_check(fs=2.0, fsig=1.0):
    return (fs /2 < fsig)

def get_alias_freq(fs=1.9, fsig = 1.0):
    n = 0
    falias = fsig
    fnyq = fs / 2 
    while falias >= fnyq:
        n += 1
        falias = np.abs(fsig - n * fs)
    return falias    


def quantization_with_option(signal, levels, min_range=0, max_range=1, normalized_type=None, quantz_type='manual'):
    switcher = {
        'manual': lambda: quantization(signal, levels, min_range, max_range, normalized_type)
    }
    return switcher.get(quantz_type, lambda: quantization(signal, levels, min_range, max_range, normalized_type))()

def measure_energy(signal):
    return np.mean(signal**2)

def sqnr_measure(normalized_signal, quantz_signal):

    noise = normalized_signal - quantz_signal  

    # tranh loi chia cho 0
    sig_noise_ratio = measure_energy(normalized_signal)/(measure_energy(noise) + 1e-6)
    return 10 * np.log10(sig_noise_ratio)


def showcase1_exp(signal_list, duration=1.0, sampling_rate=50.0, superposition=False, components=True):
    
    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    t = np.linspace(0, duration, int(duration * sampling_rate))

    fig, ax = plt.subplots(figsize=(15, 5))
    tot_sig = 0
    for ind, src in enumerate(signal_list):
        sig_type = src['signal_type']
        amp = src['amplitude']
        freq = src['frequency']
        t_sh = src['time_shift']
        
        # Thêm amplitude vào hàm gen_continous_signal
        cont_sig = gen_continous_signal(t, time_shift=t_sh, freq=freq, amplitude=amp, signal_type=sig_type)
        
        if components: 
            color = color_list[ind % len(color_list)]
            ax.plot(t, cont_sig, color=color, label=f'source[{ind}]: {sig_type}')
        tot_sig += cont_sig

    if superposition:
        ax.plot(t, tot_sig, color='black', label=f'total signal')

    ax.set_title('Biểu diễn nhiều tín hiệu liên tục và tín hiệu tổng hợp')
    ax.set_xlabel('thời gian (giây)')
    ax.set_ylabel('Biên độ')
    ax.legend(loc='upper right')

    plt.show()


def showcase3(signal_list, duration=1.0, sampling_rate=50.0, superposition=False, components=True):
    
    
    color_list = ['r', 'b', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Các mẫu về thời gian
    t_discrete = np.linspace(0, duration, int(duration * sampling_rate), endpoint= False)

    # Vì là mô phỏng nên tín hiệu liên tục được giả lập bởi tín hiệu rời rạc 
    # với tốc độ lấy mẫu gấp nhiều lần so với tín hiệu rời rạc 
    rate_continous = sampling_rate * 100
    t_cont= np.linspace(0, duration, int(duration * rate_continous),endpoint=False)

    fig, ax = plt.subplots(len(signal_list)+1, 1, figsize=(15, 8))

    if len(signal_list)==0:
        return
    
    # tín hiệu tổng hợp: liên tục và rời rạc
    tot_sig_cont = 0
    tot_sig_dist = 0

    for ind, src in enumerate(signal_list):
        sig_type = src['signal_type']
        amp = src['amplitude']
        freq = src['frequency']
        t_sh = src['time_shift']
        
        # Thêm amplitude vào hàm gen_continous_signal
        cont_sig = gen_continous_signal(t_cont, time_shift=t_sh, freq=freq, amplitude=amp, signal_type=sig_type)
        discrete_sig = gen_continous_signal(t_discrete, time_shift=t_sh, freq=freq, amplitude=amp, signal_type=sig_type)
        
        if components: 
            color = color_list[ind % len(color_list)]
            ax[ind].plot(t_cont, cont_sig, color=color, label=f'source[{ind}]: {sig_type}')
            ax[ind].stem(t_discrete, discrete_sig, linefmt=f'{color}-', markerfmt=f'{color}o', 
                         basefmt="k-", label=f'source[{ind}]: {sig_type}')  
            ax[ind].set_title(f'Biểu diễn tín hiệu liên tục & lấy mẫu source[{ind}]: {sig_type}')
            ax[ind].set_xlabel('thời gian (giây)')
            ax[ind].set_ylabel('Biên độ')
            ax[ind].legend(loc='upper right')

        tot_sig_cont += cont_sig
        tot_sig_dist += discrete_sig

    if superposition:
        ax[len(signal_list)].plot(t_cont, tot_sig_cont, color='black', label=f'total signal')
        ax[len(signal_list)].stem(t_discrete, tot_sig_dist, linefmt='k-', markerfmt='ko', 
                                  basefmt="k-", label=f'total signal')
        ax[len(signal_list)].set_title(f'Biểu diễn tín hiệu tổng hợp liên tục & lấy mẫu')
        ax[len(signal_list)].set_xlabel('thời gian (giây)')
        ax[len(signal_list)].set_ylabel('Biên độ')
        ax[len(signal_list)].legend(loc='upper right')


    plt.show()


def fir_filter(x, kernel):
    filtered_signal = signal.lfilter(kernel, [1.0], x)
    # filtered_signal = signal.lfilter(weights, [1.0], signal)
    return filtered_signal

def mac_filter(signal, kernel):

    filterd_value = []
    # Khởi tạo giá trị tích lũy
    accumulator = 0

    pad_amount = (len(kernel) - 1) 
    pad_amount = 0 

    padded_signal = np.pad(signal, (pad_amount, pad_amount), 'constant')  # Padding cho 'same'


    # Khởi tạo một buffer có độ dài bằng kernel
    buffer = np.zeros(len(kernel))
    
    # Duyệt qua từng giá trị của tín hiệu đầu vào
    for i, val in enumerate(padded_signal):
        # Đẩy giá trị mới vào buffer (mô phỏng trượt của kernel)
        buffer = np.roll(buffer, 1)
        buffer[0] = val  # Thêm giá trị mới nhất vào buffer

        # print(f'* inputed {i}-th sample')
        # print(f'    - buffer = {buffer}')
        # print(f'    - kernel = {kernel}')
        
        # Reset lại giá trị tích lũy
        accumulator = 0
        
        # Thực hiện tính MAC: nhân từng phần tử trong buffer với kernel và cộng dần vào accumulator
        for b, k in zip(buffer, kernel):
            accumulator += b * k
        
        # Yield kết quả (từng giá trị được tính từ bộ MAC)
        # print(f'    - MAC output = {accumulator}')
        # yield accumulator

        filterd_value.append(accumulator)
        
    return filterd_value


def ideal_lowpass(kernelsize=1024, fc=10):
    
    w = np.zeros(kernelsize)
    if kernelsize%2 ==1:
        mid = (kernelsize-1)//2
    else:
        mid = kernelsize//2

    for n in range(mid+1):            
        if mid + n <= kernelsize-1:
            w[mid+n] = np.sinc(2 * fc * n / (kernelsize-1))
        if mid -n >=0:
            w[mid-n] = np.sinc(2 * fc * n / (kernelsize-1))
    
    return w