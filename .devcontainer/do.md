# Build from Docker

使用vscode devcontainer工具，先编写devcontainer.json

```json
{
  "name": "vLLM GPU Dev Container",
	  "image": "vllm/vllm-openai:latest",
  "privileged": true,
  "runArgs": [
    "--gpus=all",
    "--shm-size=16g",
    "--ulimit", "memlock=-1",
    "--ulimit", "stack=67108864",
  ],
  "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind",
  "workspaceFolder": "/workspace",
  "remoteUser": "root"
}

```

Ctlr+p+shift: 选择 Rebuild Container

```
apt update
apt install -y git
apt install -y cuda-toolkit-12



git clone https://github.com/vllm-project/vllm.git
cd vllm
python3.12 -m venv .venv
source .venv/bin/activate
python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['build-system']['requires']))" > build_requirements.txt
pip install -r build_requirements.txt
pip install -r requirements/cuda.txt

```

设置cmake 的build 目录：在setup.py中的class cmake_build_ext(build_ext)添加如下的finalize_options()函数。这个的作用是设置cmake的build目录为一个固定的目录。否则，每次pip install -e . 都会使用一个不同的临时目录。

```
class cmake_build_ext(build_ext):
    def finalize_options(self):
        super().finalize_options()
        self.build_temp = os.path.join(os.getcwd(), "build")
```

```python
pip install -e . --no-build-isolation -v
```