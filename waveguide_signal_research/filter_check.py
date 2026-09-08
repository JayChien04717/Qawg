"""Numerical DSP validation only; no hardware connection or quantum-state inference."""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from QAWG.alazar import design_lowpass_fir, downconvert, fir_lowpass_iq

root = Path(__file__).resolve().parent
fs = 1e9
designs = {'Previous: cutoff 20 MHz, 101 taps': design_lowpass_fir(fs,20e6,101,8),
           'Example: pass 0-30 MHz, cutoff 50 MHz, 181 taps': design_lowpass_fir(fs,50e6,181,8)}
freq = np.fft.rfftfreq(131072, 1/fs)
fig, ax = plt.subplots(figsize=(10,5))
report = {}
for name,h in designs.items():
    response = abs(np.fft.rfft(h,131072))
    db = 20*np.log10(np.maximum(response,1e-15))
    ax.plot(freq/1e6,db,label=name)
    # Whole-band extrema, not isolated stop-band zeros.
    report[name] = {'max_abs_passband_error_db_0_30MHz':float(np.max(abs(db[freq<=30e6]))),
                    'max_stopband_db_70MHz_to_Nyquist':float(np.max(db[freq>=70e6])),
                    'edge_samples_each_side':(len(h)-1)//2}
    if len(h)==181:
        assert report[name]['max_abs_passband_error_db_0_30MHz'] < 0.1
        assert report[name]['max_stopband_db_70MHz_to_Nyquist'] < -60
    t=np.arange(20000)/fs
    gains={}
    for fm in [5e6,20e6,30e6]:
        envelope=1+0.2*np.cos(2*np.pi*fm*t)
        raw=(envelope*np.cos(2*np.pi*50e6*t))[None,:]
        z=fir_lowpass_iq(downconvert(raw,fs,50e6),h)[0]
        sl=slice(1000,-1000)
        matrix=np.column_stack([np.ones(len(t[sl])),np.cos(2*np.pi*fm*t[sl]),np.sin(2*np.pi*fm*t[sl])])
        coef=np.linalg.lstsq(matrix,z[sl].real,rcond=None)[0]
        gains[str(fm)]=float(np.hypot(coef[1],coef[2])/0.2)
    report[name]['synthetic_AM_recovered_amplitude_ratio']=gains
ax.axvspan(0,30,color='green',alpha=.08,label='Example required passband')
ax.axvspan(70,150,color='grey',alpha=.1,label='Example image stopband')
ax.set(xlim=(0,150),ylim=(-120,3),xlabel='Baseband frequency (MHz)',ylabel='Amplitude response (dB)',title='50 MHz real IF: design for the required bandwidth')
ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
fig.savefig(root/'filter_response.png',dpi=160)
(root/'filter_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
