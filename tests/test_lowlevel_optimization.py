import numpy as np
import pytest
from QAWG.alazar.dsp import downconvert, fir_lowpass_iq, integrate
from QAWG.alazar import ats9371 as ats


@pytest.mark.parametrize('shape', [(3,77),(2,3,77),(0,77)])
def test_batched_fir_matches_direct_convolution_for_strided_inputs(shape):
    rng=np.random.default_rng(23)
    x=(rng.normal(size=shape)+1j*rng.normal(size=shape))[...,::2]
    h=np.array([.1,.2,.4,.2,.1])
    original=x.copy()
    actual=fir_lowpass_iq(x,h,batch_records=2)
    expected=np.empty_like(x)
    for index in np.ndindex(x.shape[:-1]):
        full=np.convolve(x[index],h,'full')
        expected[index]=full[2:2+x.shape[-1]]
    np.testing.assert_allclose(actual,expected,atol=1e-14)
    np.testing.assert_array_equal(x,original)
    assert actual.flags.owndata and actual.flags.c_contiguous


def test_fir_longer_than_trace_and_bad_batch():
    x=np.ones((3,4),complex);h=np.ones(21)/21
    actual=fir_lowpass_iq(x,h,batch_records=1)
    np.testing.assert_allclose(actual[0],np.convolve(x[0],h,'full')[10:14],atol=1e-15)
    with pytest.raises(ValueError):fir_lowpass_iq(x,h,batch_records=0)


def test_ddc_cache_isolation_and_phase_convention():
    x=np.arange(120,dtype=float).reshape(3,40)[:,::2]
    t=np.arange(x.shape[1])/1e9
    expected=2*x*np.exp(-1j*(2*np.pi*50e6*t+.27))
    y=downconvert(x,1e9,50e6,.27)
    np.testing.assert_allclose(y,expected,atol=1e-12)
    y[:]=0
    np.testing.assert_allclose(downconvert(x,1e9,50e6,.27),expected,atol=1e-12)
    with pytest.raises(ValueError):downconvert(x,float('nan'),50e6)


def test_weighted_integration_nonconjugating_strided_multidimensional():
    rng=np.random.default_rng(90)
    x=(rng.normal(size=(2,3,20))+1j*rng.normal(size=(2,3,20)))[...,::2]
    w=np.arange(8)+1j*np.arange(8)[::-1]
    np.testing.assert_allclose(integrate(x,1,9,w),np.sum(x[...,1:9]*w,axis=-1)/sum(abs(w)),atol=1e-14)


def test_free_capture_attempts_all_and_only_retries_failed_allocations(monkeypatch):
    session=ats.CaptureSession(1,[ats.DmaBuffer(i,1,2) for i in (10,20,30)],1,1,1,1,12,active=False)
    calls=[]
    def release(address):
        calls.append(address)
        if address==20 and calls.count(20)==1:raise RuntimeError('OS release failed')
    monkeypatch.setattr(ats,'release_dma_memory',release)
    with pytest.raises(RuntimeError):ats.free_capture(session)
    assert calls==[10,20,30] and [b.address for b in session.buffers]==[20]
    ats.free_capture(session);ats.free_capture(session)
    assert calls==[10,20,30,20] and session.released


def test_reject_double_start_and_released_session():
    class API:
        def start_capture(self,handle):pass
    session=ats.CaptureSession(1,[],1,1,1,1,12)
    ats.start_capture(API(),session)
    with pytest.raises(RuntimeError):ats.start_capture(API(),session)
    session.started=False;session.released=True
    with pytest.raises(RuntimeError):ats.start_capture(API(),session)
