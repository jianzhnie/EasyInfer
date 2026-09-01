import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    "/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat"
)


def _run_with_fake_vllm(tmp_path: Path, script: str) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "capture.json"
    fake_vllm = bin_dir / "vllm"
    fake_vllm.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "keys = ['VLLM_LONGCAT_PATCH', 'VLLM_LONGCAT_DISABLE_FUSED_GATING', "
        "'EASYINFER_MOE_COMM']\n"
        "json.dump({'argv': sys.argv[1:], 'env': {k: os.environ.get(k) for k "
        "in keys}}, open(os.environ['CAPTURE_FILE'], 'w'))\n",
        encoding="utf-8",
    )
    fake_vllm.chmod(0o755)
    fake_ray = bin_dir / "ray"
    fake_ray.write_text("#!/bin/sh\necho '64.0/64.0 NPU'\n", encoding="utf-8")
    fake_ray.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "CAPTURE_FILE": str(capture),
            "LOG_FILE": str(tmp_path / "run.log"),
            "MODEL_PATH": str(MODEL_PATH),
            "PORT": "8010",
            "RAY_ADDRESS": "10.16.201.229:6379",
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    subprocess.run(
        ["bash", str(REPO_ROOT / script)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return json.loads(capture.read_text(encoding="utf-8"))


def _arg_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_default_script_enables_per_process_longcat_plugin(tmp_path):
    result = _run_with_fake_vllm(
        tmp_path, "examples/longcat/vllm/run_vllm.sh"
    )

    assert result["env"] == {
        "VLLM_LONGCAT_PATCH": "1",
        "VLLM_LONGCAT_DISABLE_FUSED_GATING": "1",
        "EASYINFER_MOE_COMM": "allgather",
    }
    assert result["argv"][0] == "serve"
    assert "--enable-expert-parallel" in result["argv"]
    assert _arg_value(result["argv"], "--tensor-parallel-size") == "32"
    assert _arg_value(result["argv"], "--pipeline-parallel-size") == "2"


def test_documented_default_port_is_consistent():
    run_script = (REPO_ROOT / "examples/longcat/vllm/run_vllm.sh").read_text()
    test_script = (REPO_ROOT / "examples/longcat/vllm/curl_test.sh").read_text()
    readme = (REPO_ROOT / "examples/longcat/vllm/README.md").read_text()

    assert 'PORT="${PORT:-8010}"' in run_script
    assert 'PORT="${PORT:-8010}"' in test_script
    assert "端口: **8010**" in readme


def test_long_context_script_matches_readme_defaults(tmp_path):
    result = _run_with_fake_vllm(
        tmp_path, "examples/longcat/vllm/run_vllm_long-context.sh"
    )

    assert _arg_value(result["argv"], "--max-model-len") == "131072"
    assert _arg_value(result["argv"], "--max-num-seqs") == "32"
    assert _arg_value(result["argv"], "--max-num-batched-tokens") == "16384"
    assert "--enable-chunked-prefill" in result["argv"]
    assert _arg_value(result["argv"], "--kv-cache-dtype") == "bfloat16"
