from h3_icr.backend import BackendDescriptor, descriptor_from_model, tag_model


class DummyModel:
    def __init__(self):
        self.model_options = {"transformer_options": {"keep": 1}}

    def clone(self):
        other = DummyModel()
        other.model_options = {
            "transformer_options": dict(self.model_options.get("transformer_options", {}))
        }
        return other


def test_backend_tag_does_not_mutate_source():
    model = DummyModel()
    desc = BackendDescriptor(kind="hybrid_late_adaln", checkpoint_format="pruned")
    tagged = tag_model(model, desc)
    assert "h3_icr_backend" not in model.model_options["transformer_options"]
    assert descriptor_from_model(tagged).kind == "hybrid_late_adaln"
    assert tagged.model_options["transformer_options"]["keep"] == 1
