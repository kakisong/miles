import json
import logging
import os
import tempfile
from contextlib import contextmanager

try:
    from sglang.srt.utils.hf_transformers_utils import _load_deepseek_temp_model
except ImportError:
    # Older sglang (radixark/miles:latest x86 ships nightly 20260103) lacks the
    # parametrised loader; fall back to a local shim with the same shape as upstream.
    from transformers.models.auto.configuration_auto import AutoConfig

    def _load_deepseek_temp_model(
        model_path: str,
        model_type: str = "deepseek_v3",
        architecture: str = "DeepseekV3ForCausalLM",
        trust_remote_code: bool = False,
        revision=None,
        **kwargs,
    ):
        config_file = os.path.join(model_path, "config.json")
        if not os.path.exists(config_file):
            raise RuntimeError(f"Can't find config file in {model_path}.")
        with open(config_file) as f:
            cfg = json.load(f)
        cfg["architectures"] = [architecture]
        cfg["model_type"] = model_type
        tmp_dir = os.path.join(tempfile.gettempdir(), "_miles_dsv4_cfg")
        unique = os.path.join(tmp_dir, f"deepseek_temp_{os.getpid()}")
        os.makedirs(unique, exist_ok=True)
        with open(os.path.join(unique, "config.json"), "w") as f:
            json.dump(cfg, f)
        # Bypass our own AutoConfig.from_pretrained patch (would recurse since
        # the rewritten config still has model_type="deepseek_v3"). Use the
        # saved original if patch is active, else current AutoConfig.from_pretrained.
        loader = _original_from_pretrained or AutoConfig.from_pretrained
        if hasattr(loader, "__func__"):
            loaded = loader.__func__(
                AutoConfig, unique, trust_remote_code=trust_remote_code, revision=revision, **kwargs
            )
        else:
            loaded = loader(unique, trust_remote_code=trust_remote_code, revision=revision, **kwargs)
        # transformers 5.3 V3Config consolidates legacy fields like rope_theta into
        # rope_parameters dict and drops V4-only fields. Re-attach raw config.json
        # entries so callers reading e.g. hf_config.rope_theta still work.
        for k, v in cfg.items():
            if not hasattr(loaded, k):
                try:
                    setattr(loaded, k, v)
                except Exception:
                    pass
        # V4 config.json doesn't include `first_k_dense_replace` / `intermediate_size`
        # because V4-Flash is fully MoE. V3Config defaults to first_k_dense_replace=3
        # which would mislead mbridge into building 3 dense MLP layers and trying to
        # load nonexistent gate_proj/up_proj weights. Force V4 topology defaults:
        v4_defaults = {
            "first_k_dense_replace": 0,
            "intermediate_size": cfg.get("moe_intermediate_size", 2048),
        }
        for k, v in v4_defaults.items():
            try:
                setattr(loaded, k, v)
            except Exception:
                pass
        return loaded


logger = logging.getLogger(__name__)

_original_from_pretrained = None


@contextmanager
def with_transformers_patch():
    apply_transformers_patch()
    try:
        yield
    finally:
        unapply_transformers_patch()


def apply_transformers_patch():
    global _original_from_pretrained
    if _original_from_pretrained is not None:
        return

    from transformers.models.auto.configuration_auto import AutoConfig

    _original_from_pretrained = AutoConfig.from_pretrained

    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers.configuration_utils import PretrainedConfig

        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get("model_type") in ("deepseek_v4", "deepseek_ref", "deepseek_v3"):
            return _load_deepseek_temp_model(
                pretrained_model_name_or_path,
                model_type="deepseek_v3",
                architecture="DeepseekV3ForCausalLM",
                **kwargs,
            )

        return _original_from_pretrained.__func__(cls, pretrained_model_name_or_path, **kwargs)

    AutoConfig.from_pretrained = _patched_from_pretrained


def unapply_transformers_patch():
    global _original_from_pretrained
    if _original_from_pretrained is None:
        return

    from transformers.models.auto.configuration_auto import AutoConfig

    AutoConfig.from_pretrained = _original_from_pretrained
    _original_from_pretrained = None
