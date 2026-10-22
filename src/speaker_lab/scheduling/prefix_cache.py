"""Connection-local clean LM prefix; never replace CUDA graph state tensors.

The caller captures this after the fixed voice/persona prefix, before any tool
or user facts. It remains in GPU memory and is discarded with the session.
Torch is deliberately supplied by the backend, not a gateway dependency.
"""

from dataclasses import fields, is_dataclass


class StreamingPrefix:
    def __init__(self, module, tensor_type):
        self.module = module
        self.states = dict(module.get_streaming_state())
        self.tensors = []
        self.scalars = []
        self.graphs = []
        self.objects = []
        graph_fields = {"graphed_main", "graphed_embeddings", "graphed_depth"}

        def capture(obj, root=False):
            if is_dataclass(obj):
                names = [field.name for field in fields(obj)]
            elif hasattr(obj, "asdict"):
                names = list(obj.asdict())
            else:
                raise TypeError(f"Unsupported prefix state: {type(obj).__name__}")
            for name in names:
                value = getattr(obj, name)
                if root and name in graph_fields:
                    self.graphs.append((obj, name, value))
                elif isinstance(value, tensor_type):
                    self.tensors.append((obj, name, value, value.clone()))
                elif isinstance(value, (int, float, str, bool, type(None))):
                    self.scalars.append((obj, name, value))
                else:
                    self.objects.append((obj, name, value))
                    capture(value)

        for name, state in self.states.items():
            capture(state, root=name == "")

    def restore(self):
        current = self.module.get_streaming_state()
        if current.keys() != self.states.keys() or any(
            current[name] is not state for name, state in self.states.items()
        ):
            raise RuntimeError("Streaming prefix state topology changed")
        # Validate all references before mutating anything. Graphs capture the
        # existing tensor addresses; swapping in cloned states is unsafe.
        for owner, name, value in self.graphs + self.objects:
            if getattr(owner, name) is not value:
                raise RuntimeError("Streaming prefix graph changed")
        for owner, name, value, _ in self.tensors:
            if getattr(owner, name) is not value:
                raise RuntimeError("Streaming prefix tensor changed")
        for _, _, value, saved in self.tensors:
            value.copy_(saved)
        for owner, name, value in self.scalars:
            setattr(owner, name, value)
