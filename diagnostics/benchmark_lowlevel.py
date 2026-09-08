"""Offline benchmark against the previous implementations; no hardware access."""
from pathlib import Path
import sys
import time
import tracemalloc
import gc
import json
import platform
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from QAWG.alazar.dsp import downconvert, fir_lowpass_iq, design_lowpass_fir

def old_ddc(x):
    t=np.arange(x.shape[1])/1e9
    return 2*x*np.exp(-1j*(2*np.pi*50e6*t))

def old_fir(x,h):
    size=x.shape[-1]+len(h)-1
    n=1<<(size-1).bit_length()
    full=np.fft.ifft(np.fft.fft(x,n=n,axis=-1)*np.fft.fft(h,n=n),axis=-1)[...,:size]
    d=(len(h)-1)//2
    return full[...,d:d+x.shape[-1]]

def measure(fn):
    value=fn();del value
    times=[]
    for _ in range(5):
        gc.collect();start=time.perf_counter();value=fn()
        times.append(time.perf_counter()-start);del value
    gc.collect();tracemalloc.start();value=fn()
    retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    owner=value
    while isinstance(owner.base,np.ndarray):owner=owner.base
    result={'median_seconds':float(np.median(times)),
            'traced_peak_MiB':peak/2**20,'traced_retained_MiB':retained/2**20,
            'output_backing_MiB':owner.nbytes/2**20}
    del value
    return result

def main():
    rng=np.random.default_rng(1);raw=rng.normal(size=(1024,2048))
    iq=old_ddc(raw);h=design_lowpass_fir(1e9,50e6,181,8)
    np.testing.assert_allclose(downconvert(raw,1e9,50e6),iq,atol=1e-11)
    np.testing.assert_allclose(fir_lowpass_iq(iq,h),old_fir(iq,h),atol=1e-12)
    report={'python':platform.python_version(),'numpy':np.__version__,'shape':list(raw.shape),
            'notes':'5-run medians after warmup; tracemalloc excludes pre-existing input and may not capture all native allocations; not ADC throughput'}
    for name,fn in [('old_ddc',lambda:old_ddc(raw)),('new_ddc',lambda:downconvert(raw,1e9,50e6)),
                    ('old_fir',lambda:old_fir(iq,h)),('new_fir',lambda:fir_lowpass_iq(iq,h))]:
        report[name]=measure(fn)
    path=Path(__file__).with_name('lowlevel_benchmark.json')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
