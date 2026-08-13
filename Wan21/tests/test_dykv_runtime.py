import importlib.util
import pathlib
import sys
import types
import unittest

import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
PIPELINE_ROOT = WAN21_ROOT / "pipeline"
package = types.ModuleType("dykv_test_pipeline")
package.__path__ = [str(PIPELINE_ROOT)]
sys.modules[package.__name__] = package


def _load(name):
    full_name = f"{package.__name__}.{name}"
    spec = importlib.util.spec_from_file_location(full_name, PIPELINE_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


_load("dykv_fov")
memory = _load("dykv_memory")
runtime_module = _load("dykv_runtime")


def _w2c(x):
    c2w = torch.eye(4)
    c2w[0, 3] = float(x)
    return torch.linalg.inv(c2w)


def _cache(value, tokens=4):
    key = torch.full((1, tokens, 1, 2), float(value))
    return {
        "k": key,
        "v": key + 100,
        "local_end_index": torch.tensor(tokens),
    }


class DyKVRuntimeTest(unittest.TestCase):
    def test_archive_select_compress_pipeline(self):
        config = memory.DyKVConfig(
            enabled=True,
            memory_frames=8,
            fov_samples=2048,
            compression_keep_ratio=0.5,
        )
        runtime = runtime_module.DyKVRuntime(config, chunk_frames=4)
        for start, x in zip((0, 4, 8, 12, 16), (0, 1, 8, 2, 3)):
            poses = torch.stack([_w2c(x)] * 4).unsqueeze(0)
            runtime.archive(
                "main",
                [_cache(start)],
                frame_start=start,
                frame_count=4,
                frame_tokens=1,
                viewmats=poses,
            )

        current = torch.stack([_w2c(1.1)] * 4).unsqueeze(0)
        payloads = runtime.retrieve(
            "main",
            current_frame=20,
            current_viewmats=current,
            frame_tokens=1,
            target_device="cpu",
        )

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["src_frame_ids"], [4, 12])
        self.assertEqual(payloads[0]["k"].shape[1], 8)
        event = runtime.summary()["events"][0]
        self.assertEqual(event["selected_frame_starts"], [4, 12])
        self.assertEqual(event["retrieved_tokens_per_layer"], 8)


if __name__ == "__main__":
    unittest.main()
