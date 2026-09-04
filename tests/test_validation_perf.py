from types import SimpleNamespace

import torch

from h3_icr.validation_perf import (
    SamplerPerformanceStats,
    sampler_performance_wrapper,
)


def test_performance_stats_separate_first_use_from_steady_state():
    stats = SamplerPerformanceStats()
    stats.record(
        wall_seconds=3.0,
        cuda_device="cuda:0",
        peak_allocated_bytes=1024,
        peak_reserved_bytes=2048,
    )
    stats.record(
        wall_seconds=1.5,
        cuda_device="cuda:0",
        peak_allocated_bytes=1536,
        peak_reserved_bytes=3072,
    )
    stats.record(
        wall_seconds=1.0,
        cuda_device="cuda:0",
        peak_allocated_bytes=1280,
        peak_reserved_bytes=2560,
    )
    report = stats.to_dict()
    assert report["calls"] == 3
    assert report["first_wall_seconds"] == 3.0
    assert report["steady_wall_seconds_mean"] == 1.25
    assert report["steady_wall_seconds_min"] == 1.0
    assert report["steady_wall_seconds_max"] == 1.5
    assert report["peak_allocated_bytes_max"] == 1536
    assert report["peak_reserved_bytes_max"] == 3072


def test_sampler_performance_wrapper_is_output_neutral_on_cpu(monkeypatch):
    stats = SamplerPerformanceStats()
    transformer = {"h3_icr_validation_performance_stats": stats}
    extra_args = {"model_options": {"transformer_options": transformer}}
    expected = torch.tensor([1.0, 2.0])
    calls = []

    def executor(guider, sigmas, incoming_extra, callback, noise, *args, **kwargs):
        calls.append((guider, sigmas, incoming_extra, callback, noise, args, kwargs))
        return expected

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = sampler_performance_wrapper(
        executor,
        "guider",
        torch.tensor([0.5, 0.0]),
        extra_args,
        None,
        torch.zeros(1),
        "arg",
        flag=True,
    )
    assert output is expected
    assert len(calls) == 1
    assert stats.cpu_or_unresolved_calls == 1
    assert stats.cuda_available_calls == 0
    assert stats.to_dict()["first_wall_seconds"] is not None


def test_sampler_performance_wrapper_delegates_when_stats_are_missing():
    marker = object()

    def executor(*args, **kwargs):
        return marker

    output = sampler_performance_wrapper(
        executor,
        SimpleNamespace(),
        torch.tensor([0.5, 0.0]),
        {"model_options": {"transformer_options": {}}},
        None,
        torch.zeros(1),
    )
    assert output is marker
