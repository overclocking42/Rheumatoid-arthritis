# macOS Apple Silicon Inference Setup

This project should run locally on Apple Silicon without TensorFlow. The deployed app uses PyTorch for both numeric inference and the Swin X-ray model.

## Root cause

- The global machine environment had `tensorflow==2.20.0`, `pyarrow==21.0.0`, and `Python 3.13.7` on macOS arm64.
- Importing TensorFlow in that environment aborts with `libc++abi: ... mutex lock failed: Invalid argument`.
- The same environment also causes `transformers` to hang, because `transformers` can interact badly with a broken TensorFlow install even when you only want PyTorch inference.
- The app had a separate bug in its Swin loader: it rewrote checkpoint keys incorrectly even though the checkpoint already contains the `swin.` prefix.

## Clean inference-only environment

Use a dedicated virtual environment and do not install TensorFlow, Keras, or `tensorflow-metal`.

```bash
cd /Users/joyboy/Documents/projects/Rheumatoid-Arthritis/Rheumatoid-arthritis
/opt/homebrew/bin/python3.12 -m venv .venv-macos-infer
source .venv-macos-infer/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-macos-inference.txt
```

If `python3.12` is not installed:

```bash
brew install python@3.12
```

## Packages to avoid in this environment

```bash
python -m pip uninstall -y tensorflow tensorflow-macos tensorflow-metal keras tensorboard
```

## Safe smoke test

This verifies that both saved models load on CPU and that one sample X-ray runs end to end:

```bash
export USE_TF=0
export TRANSFORMERS_NO_TF=1
python scripts/test_local_inference.py
```

## Run the app

```bash
export USE_TF=0
export TRANSFORMERS_NO_TF=1
streamlit run src/app/app_auth.py
```

## If you still want to test TensorFlow separately

Keep it in a different virtual environment. Do not mix it into the PyTorch app environment.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-tf-test
source .venv-tf-test/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install tensorflow==2.21.0
python - <<'PY'
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices())
PY
```

If you later test Metal:

```bash
python -m pip install tensorflow-metal
```

Do that only after TensorFlow CPU import works in the separate TensorFlow-only environment.
