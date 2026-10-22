"""A clean prefix restores values without replacing CUDA graph tensor addresses."""
from dataclasses import dataclass
import pytest


class Tensor:
    def __init__(self, value):
        self.value = value
    def clone(self):
        return Tensor(self.value)
    def copy_(self, other):
        self.value = other.value


@dataclass
class Generator:
    cache: Tensor
    offset: int
    graphed_main: object
    graphed_embeddings: object
    graphed_depth: object


class Ring:
    def __init__(self):
        self.cache, self.end_offset = Tensor(10), Tensor(2)
    def asdict(self):
        return dict(cache=self.cache, end_offset=self.end_offset)


@dataclass
class Attention:
    kv_cache: Ring
    offset: Tensor
    offset_cpu: int


class Module:
    def __init__(self):
        self.states = {'': Generator(Tensor(11), 2, object(), object(), object()),
                       'attention': Attention(Ring(), Tensor(2), 2)}
    def get_streaming_state(self):
        return self.states


def test_repeated_restore_removes_old_facts_and_preserves_graph_objects_and_tensors():
    from speaker_lab.scheduling.prefix_cache import StreamingPrefix
    m = Module()
    prefix = StreamingPrefix(m, Tensor)
    original = m.states[''].cache
    graph = m.states[''].graphed_main
    kv = m.states['attention'].kv_cache.cache
    for _ in range(3):
        original.value = 999
        kv.value = 888
        m.states[''].offset = 90
        m.states['attention'].offset_cpu = 90
        m.states['attention'].offset.value = 90
        prefix.restore()
        assert original is m.states[''].cache and original.value == 11
        assert graph is m.states[''].graphed_main
        assert kv is m.states['attention'].kv_cache.cache and kv.value == 10
        assert m.states[''].offset == 2
        assert m.states['attention'].offset_cpu == 2
        assert m.states['attention'].offset.value == 2


def test_replaced_streaming_state_rejected_before_restore():
    from speaker_lab.scheduling.prefix_cache import StreamingPrefix
    m = Module()
    prefix = StreamingPrefix(m, Tensor)
    m.states[''].offset = 99
    m.states['attention'] = Attention(Ring(), Tensor(5), 5)
    with pytest.raises(RuntimeError, match='changed'):
        prefix.restore()
    assert m.states[''].offset == 99


def test_replaced_tensor_rejected_before_any_restore():
    from speaker_lab.scheduling.prefix_cache import StreamingPrefix
    m = Module()
    prefix = StreamingPrefix(m, Tensor)
    m.states[''].offset = 99
    m.states['attention'].kv_cache.cache = Tensor(777)
    with pytest.raises(RuntimeError, match='changed'):
        prefix.restore()
    assert m.states[''].offset == 99


def test_unknown_mutable_state_is_rejected_rather_than_silently_ignored():
    from speaker_lab.scheduling.prefix_cache import StreamingPrefix
    m = Module()
    m.states['unknown'] = object()
    with pytest.raises(TypeError, match='Unsupported'):
        StreamingPrefix(m, Tensor)
