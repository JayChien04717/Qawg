import ctypes
import numpy as np
import pytest

from QAWG.alazar import CaptureConfig, CaptureData, IFProcessor, reference_phase_correct, save_mode_shots
from QAWG.alazar import ats9371 as ats


def test_dual_dma_interleaving_and_single_compatibility():
    # Distinct records/channels expose swaps and buffer pairing errors.
    a=np.arange(12,dtype=np.uint16).reshape(3,4)
    b=a+1000
    packed=np.stack([a,b],axis=-1).ravel()
    memory=(ctypes.c_uint16*packed.size)(*packed)
    buffer=ats.DmaBuffer(ctypes.addressof(memory),packed.size,packed.nbytes)
    session=ats.CaptureSession(1,[buffer],4,3,3,1,12,channel_count=2)
    actual=ats.copy_dma_buffer(session,buffer)
    np.testing.assert_array_equal(actual[:,0],a)
    np.testing.assert_array_equal(actual[:,1],b)
    memory[0]=999
    assert actual[0,0,0]==0  # copied, not dangling DMA view
    session.channel_count=1
    session.samples_per_record=8
    assert ats.copy_dma_buffer(session,buffer).shape==(3,8)


def test_dual_arm_allocation_flags_and_multiple_buffers(monkeypatch):
    memory=[];calls=[]
    def allocate(n):
        m=(ctypes.c_uint16*(n//2))();memory.append(m)
        return ctypes.addressof(m)
    monkeypatch.setattr(ats,'allocate_dma_memory',allocate)
    monkeypatch.setattr(ats,'release_dma_memory',lambda address:None)
    class API:
        def __getattr__(self,name):
            return lambda *args:calls.append((name,args))
    api=API()
    cfg=CaptureConfig(256,records_per_acquisition=6,records_per_buffer=2,channel=3)
    session=ats.arm_capture(api,ats.BoardInfo(1,0,0,12),cfg)
    assert session.buffers[0].size_bytes==256*2*2*2
    args=next(args for name,args in calls if name=='before_async_read')
    assert args[1]==3 and args[-1]&0x1000
    for i,m in enumerate(memory):
        np.ctypeslib.as_array(m).reshape(2,256,2)[:,:,0]=i+1
        np.ctypeslib.as_array(m).reshape(2,256,2)[:,:,1]=i+101
    ats.start_capture(api,session)
    actual=ats.wait_for_capture(api,session,100)
    assert actual.shape==(6,2,256)
    np.testing.assert_array_equal(actual[:,0,0],[1,1,2,2,3,3])
    np.testing.assert_array_equal(actual[:,1,0],[101,101,102,102,103,103])
    ats.abort_capture(api,session);ats.free_capture(session)


def test_channel_selection():
    assert CaptureConfig(256,channel=3).channels==(1,2)
    capture=CaptureData(np.zeros((2,2,256),dtype=np.uint16),np.ones((2,2,256)),1e9,(1,2))
    assert capture.channel_records(2).shape==(2,256)
    with pytest.raises(ValueError): capture.channel_records(4)


@pytest.mark.parametrize('fail', [False, True])
def test_awg_referenced_capture_order_and_failure_cleanup(monkeypatch, fail):
    from types import SimpleNamespace
    from QAWG.awg_alazar import AWGAlazar
    import QAWG.alazar as public
    events=[]
    monkeypatch.setattr(public,'configure_channel',lambda api,handle,ch:events.append(('channel',ch)))
    obj=AWGAlazar.__new__(AWGAlazar)
    obj.alazar_sample_rate_hz=1e9
    obj._uploaded_sequence_name=None
    obj._acquisition_config=lambda n:CaptureConfig(512,records_per_acquisition=n)
    obj.awg=SimpleNamespace(stop=lambda:events.append('stop'),run=lambda **kwargs:events.append('run'))
    t=np.arange(512)/1e9
    raw=np.tile(np.cos(2*np.pi*50e6*t),(3,2,1))
    def capture(config,start_callback):
        assert config.channel==3
        events.append('armed');start_callback()
        if fail: raise RuntimeError('DMA failure')
        return CaptureData(np.zeros(raw.shape,dtype=np.uint16),raw,1e9,(1,2))
    obj.alazar=SimpleNamespace(api=None,board=SimpleNamespace(handle=1),capture=capture)
    kwargs=dict(processor=IFProcessor(),reference_window_samples=(100,300),min_reference_amplitude=.1)
    if fail:
        with pytest.raises(RuntimeError,match='DMA failure'):obj.acquire_referenced_waveforms(3,**kwargs)
    else:
        data,iq,phases=obj.acquire_referenced_waveforms(3,**kwargs)
        assert iq.shape==(3,512) and phases.shape==(3,)
    assert events[-1]=='stop'
    assert events.index('armed')<events.index('run')


def test_filter_restores_fast_envelope_and_rejects_bad_spec():
    p=IFProcessor()
    t=np.arange(8192)/1e9
    envelope=1+0.2*np.cos(2*np.pi*30e6*t)
    iq=p.process((envelope*np.cos(2*np.pi*50e6*t))[None,:])
    np.testing.assert_allclose(iq[0,200:-200].real,envelope[200:-200],atol=5e-5)
    with pytest.raises(ValueError): IFProcessor(taps=11)
    with pytest.raises(ValueError): IFProcessor(passband_hz=60e6)


def test_reference_preserves_intrinsic_phase_and_mode_roundtrip(tmp_path):
    rng=np.random.default_rng(32)
    intrinsic=rng.uniform(-np.pi,np.pi,128)
    drift=rng.uniform(-np.pi,np.pi,128)
    signal=np.exp(1j*(intrinsic+drift))[:,None]*np.ones((1,512))
    ref=2*np.exp(1j*drift)[:,None]*np.ones((1,512))
    corrected,phase=reference_phase_correct(signal,ref,(100,200),min_reference_amplitude=1)
    np.testing.assert_allclose(corrected[:,256],np.exp(1j*intrinsic),atol=1e-14)
    with pytest.raises(ValueError): reference_phase_correct(signal,ref*0,(100,200),min_reference_amplitude=1)
    weights=np.zeros(512);weights[100:400]=1/np.sqrt(300)
    path=tmp_path/'shots.hdf5'
    expected=save_mode_shots(path,iq_shots=corrected,time_s=np.arange(512)/1e9,
        weights=weights,processor=IFProcessor(),metadata={'condition':'synthetic'},reference_phase=phase)
    import h5py
    with h5py.File(path,'r') as f:
        np.testing.assert_allclose(f['mode_iq'][:],np.sqrt(300)*np.exp(1j*intrinsic),atol=1e-12)
        np.testing.assert_array_equal(f['mode_iq'][:],expected)
    with pytest.raises(FileExistsError): save_mode_shots(path,iq_shots=corrected,time_s=np.arange(512)/1e9,
        weights=weights,processor=IFProcessor(),metadata={})
    with pytest.raises(ValueError): save_mode_shots(tmp_path/'bad.h5',iq_shots=corrected,time_s=np.arange(512)/1e9,
        weights=np.ones(512)/np.sqrt(512),processor=IFProcessor(),metadata={})
