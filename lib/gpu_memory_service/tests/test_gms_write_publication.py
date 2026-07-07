# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deferred GMS write publication in the vLLM integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH, HAS_VLLM

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)

import torch
from gpu_memory_service.integrations.common import utils as common_utils
from gpu_memory_service.integrations.vllm import model_loader

if HAS_VLLM:
    from gpu_memory_service.integrations.vllm import worker as worker_module
else:
    worker_module = None

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


class _FakeAllocator:
    def __init__(self, events: list[str], *, fail_commit: bool = False):
        self.events = events
        self.fail_commit = fail_commit
        self.total_bytes = 100
        self.mappings = {
            1: SimpleNamespace(allocation_id="weight", aligned_size=60),
            2: SimpleNamespace(allocation_id="scratch", aligned_size=40),
        }

    def commit(self):
        self.events.append("commit")
        if self.fail_commit:
            raise RuntimeError("commit failed")

    def connect(self, lock_type):
        self.events.append("connect")

    def remap_all_vas(self):
        self.events.append("remap")

    def close(self, *, best_effort=False):
        self.events.append("close_best_effort" if best_effort else "close")


@pytest.fixture
def prepared_write(monkeypatch):
    events = []
    allocator = _FakeAllocator(events)

    def register(_allocator, _model):
        events.append("register")
        return {"weight"}

    def prune(_allocator, *, referenced_allocation_ids):
        assert referenced_allocation_ids == {"weight"}
        events.append("prune")
        _allocator.mappings.pop(2)
        _allocator.total_bytes = 60

    def rebind(_allocator, _model, *, retain_gms_tensors=None):
        events.append("rebind")
        if retain_gms_tensors is not None:
            retain_gms_tensors.append(object())
        return 12

    monkeypatch.setattr(common_utils, "register_module_tensors", register)
    monkeypatch.setattr(common_utils, "prune_allocations", prune)
    monkeypatch.setattr(common_utils, "rebind_nonparameter_tensors", rebind)

    prepared = common_utils.prepare_gms_write(allocator, object())
    return prepared, events


@pytest.fixture(autouse=True)
def clear_pending_write(monkeypatch):
    monkeypatch.setattr(model_loader, "_pending_gms_write", None)
    monkeypatch.setattr(model_loader, "_last_imported_weights_bytes", 0)
    monkeypatch.setattr(model_loader, "_last_model_memory_usage_offset_bytes", 0)


@pytest.mark.none
def test_prepare_registers_and_prunes_without_commit(prepared_write):
    prepared, events = prepared_write

    assert events == ["register", "prune"]
    assert prepared.stats.committed_bytes == 60
    assert prepared.stats.pruned_bytes == 40
    assert prepared.pruned_count == 1


@pytest.mark.none
def test_publish_commits_reconnects_and_remaps(prepared_write):
    prepared, events = prepared_write
    prepared.rebind_nonparameter_tensors()
    assert len(prepared._retained_gms_tensors) == 1

    stats = prepared.publish()

    assert stats.committed_bytes == 60
    assert events == ["register", "prune", "rebind", "commit", "connect", "remap"]
    assert prepared._retained_gms_tensors == []


@pytest.mark.none
def test_pending_write_publish_clears_pending(prepared_write):
    prepared, events = prepared_write
    model_loader._store_pending_gms_write(prepared)
    assert model_loader.has_pending_gms_write()

    assert model_loader.publish_pending_gms_write()

    assert events == ["register", "prune", "commit", "connect", "remap"]
    assert not model_loader.has_pending_gms_write()
    assert not model_loader.publish_pending_gms_write()


@pytest.mark.none
def test_pending_write_abort_releases_writer_without_cuda_cleanup(prepared_write):
    prepared, events = prepared_write
    model_loader._store_pending_gms_write(prepared)

    assert model_loader.abort_pending_gms_write()

    assert events == ["register", "prune", "close_best_effort"]
    assert "close" not in events
    assert not model_loader.has_pending_gms_write()
    assert not model_loader.abort_pending_gms_write()


@pytest.mark.none
def test_publication_failure_releases_writer_and_preserves_error(monkeypatch):
    events = []
    allocator = _FakeAllocator(events, fail_commit=True)
    monkeypatch.setattr(
        common_utils,
        "register_module_tensors",
        lambda _allocator, _model: {"weight"},
    )
    monkeypatch.setattr(
        common_utils,
        "prune_allocations",
        lambda _allocator, **_kwargs: None,
    )
    prepared = common_utils.prepare_gms_write(allocator, object())
    model_loader._store_pending_gms_write(prepared)

    with pytest.raises(RuntimeError, match="commit failed"):
        model_loader.publish_pending_gms_write()

    assert events == ["commit", "close_best_effort"]
    assert not model_loader.has_pending_gms_write()


@pytest.mark.none
def test_pending_write_accounting_includes_private_rebound_bytes(prepared_write):
    prepared, _ = prepared_write
    prepared.rebind_nonparameter_tensors()

    model_loader._store_pending_gms_write(prepared)

    assert model_loader.get_imported_weights_bytes() == 60
    assert model_loader.get_model_memory_usage_offset_bytes() == 52


@pytest.mark.none
def test_second_pending_write_is_rejected(prepared_write):
    prepared, _ = prepared_write
    model_loader._store_pending_gms_write(prepared)

    with pytest.raises(RuntimeError, match="already awaiting publication"):
        model_loader._store_pending_gms_write(prepared)


@pytest.mark.none
def test_eager_finalize_preserves_publish_then_rebind_order(monkeypatch):
    """SGLang/TRT-LLM keep the eager register->commit->reconnect->rebind flow."""
    events = []
    allocator = _FakeAllocator(events)

    monkeypatch.setattr(
        common_utils,
        "register_module_tensors",
        lambda _allocator, _model: events.append("register") or {"weight"},
    )

    def prune(_allocator, *, referenced_allocation_ids):
        assert referenced_allocation_ids == {"weight"}
        events.append("prune")
        _allocator.mappings.pop(2)
        _allocator.total_bytes = 60

    monkeypatch.setattr(common_utils, "prune_allocations", prune)
    monkeypatch.setattr(
        common_utils,
        "rebind_nonparameter_tensors",
        lambda _allocator, _model, **_kwargs: events.append("rebind") or 12,
    )

    stats = common_utils.finalize_gms_write(allocator, object())

    assert stats.committed_bytes == 60
    assert stats.pruned_bytes == 40
    assert events == [
        "register",
        "prune",
        "commit",
        "connect",
        "remap",
        "rebind",
    ]


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is required for worker tests")
@pytest.mark.vllm
def test_worker_profiles_before_publication(prepared_write, monkeypatch):
    assert worker_module is not None
    prepared, events = prepared_write
    model_loader._store_pending_gms_write(prepared)
    worker = object.__new__(worker_module.GMSWorker)
    monkeypatch.setattr(
        worker_module.GMSWorker,
        "_determine_available_memory_before_gms_publish",
        lambda _self: events.append("profile") or 123,
    )

    assert worker.determine_available_memory() == 123
    assert events == ["register", "prune", "profile", "commit", "connect", "remap"]
    assert not model_loader.has_pending_gms_write()


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is required for worker tests")
@pytest.mark.vllm
def test_worker_profile_failure_aborts_pending(prepared_write, monkeypatch):
    assert worker_module is not None
    prepared, events = prepared_write
    model_loader._store_pending_gms_write(prepared)
    worker = object.__new__(worker_module.GMSWorker)

    def fail_profile(_self):
        events.append("profile")
        raise ValueError("profile failed")

    monkeypatch.setattr(
        worker_module.GMSWorker,
        "_determine_available_memory_before_gms_publish",
        fail_profile,
    )

    with pytest.raises(ValueError, match="profile failed"):
        worker.determine_available_memory()

    assert events == ["register", "prune", "profile", "close_best_effort"]
    assert not model_loader.has_pending_gms_write()


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is required for worker tests")
@pytest.mark.vllm
def test_worker_memory_accounting_uses_rebound_offset(prepared_write, monkeypatch):
    assert worker_module is not None
    prepared, _ = prepared_write
    prepared.rebind_nonparameter_tensors()
    model_loader._store_pending_gms_write(prepared)
    worker = object.__new__(worker_module.GMSWorker)
    worker.model_runner = SimpleNamespace(model_memory_usage=0)
    monkeypatch.setattr(worker_module.Worker, "load_model", lambda _self: None)

    worker.load_model()

    assert worker.model_runner.model_memory_usage == 112


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is required for worker tests")
@pytest.mark.vllm
def test_pending_write_scratch_peak_counts_visible_memory_once(
    prepared_write, monkeypatch
):
    assert worker_module is not None
    from vllm.config import CUDAGraphMode

    prepared, _ = prepared_write
    prepared.rebind_nonparameter_tensors()
    model_loader._store_pending_gms_write(prepared)
    worker = object.__new__(worker_module.GMSWorker)
    worker.requested_memory = 1_000
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    )
    worker.model_runner = SimpleNamespace(
        model_memory_usage=0,
        profile_run=lambda: None,
    )
    monkeypatch.setattr(worker_module, "is_scratch_kv_enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    # Absolute peak = 60 committed + 12 private rebound + 20 profile bytes,
    # all visible to PyTorch stats while the write is pending.
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 92)

    assert worker.determine_available_memory() == 908
    assert worker.available_kv_cache_memory_bytes == 908
    assert not model_loader.has_pending_gms_write()


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is required for worker tests")
@pytest.mark.vllm
def test_ro_scratch_peak_adds_invisible_imported_weights_once(monkeypatch):
    assert worker_module is not None
    from vllm.config import CUDAGraphMode

    monkeypatch.setattr(model_loader, "_last_imported_weights_bytes", 60)
    worker = object.__new__(worker_module.GMSWorker)
    worker.requested_memory = 1_000
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    )
    worker.model_runner = SimpleNamespace(
        model_memory_usage=60,
        profile_run=lambda: None,
    )
    monkeypatch.setattr(worker_module, "is_scratch_kv_enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    # The 20-byte absolute PyTorch peak excludes the 60-byte RO GMS mapping.
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 20)

    assert not model_loader.has_pending_gms_write()
    assert worker.determine_available_memory() == 920
    assert worker.available_kv_cache_memory_bytes == 920


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is required for worker tests")
@pytest.mark.vllm
def test_initialize_publishes_before_connector_and_kv_allocation(
    prepared_write, monkeypatch
):
    assert worker_module is not None
    from vllm.distributed import kv_transfer

    prepared, events = prepared_write
    model_loader._store_pending_gms_write(prepared)
    worker = object.__new__(worker_module.GMSWorker)
    worker.local_rank = 0
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=False)
    )
    worker.model_runner = SimpleNamespace(
        initialize_kv_cache=lambda _config: events.append("initialize")
    )
    monkeypatch.setattr(worker_module, "is_scratch_kv_enabled", lambda: False)
    monkeypatch.setattr(
        kv_transfer,
        "ensure_kv_transfer_initialized",
        lambda *_args: events.append("connector"),
    )

    worker.initialize_from_config(object())

    assert events == [
        "register",
        "prune",
        "commit",
        "connect",
        "remap",
        "connector",
        "initialize",
    ]


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is required for worker tests")
@pytest.mark.vllm
def test_initialize_without_pending_write_is_noop(monkeypatch):
    assert worker_module is not None
    from vllm.distributed import kv_transfer

    events = []
    worker = object.__new__(worker_module.GMSWorker)
    worker.local_rank = 0
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=False)
    )
    worker.model_runner = SimpleNamespace(
        initialize_kv_cache=lambda _config: events.append("initialize")
    )
    monkeypatch.setattr(worker_module, "is_scratch_kv_enabled", lambda: False)
    monkeypatch.setattr(
        kv_transfer,
        "ensure_kv_transfer_initialized",
        lambda *_args: events.append("connector"),
    )

    worker.initialize_from_config(object())

    assert events == ["connector", "initialize"]
