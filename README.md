# easybuild-llm

Integration of Large Language Models (LLMs) in [EasyBuild](https://easybuild.io).

## Features

- Ask an LLM to explain why a shell command failed, by passing a `RunShellCmdResult` instance to `explain_failed_shell_cmd`

## Design

`easybuid-llm` is implemented such that it doesn't require any changes to EasyBuild.

It leverages EasyBuild's support for using [hooks](https://docs.easybuild.io/hooks) to customize the behaviour of EasyBuild.

## Installation

### Requirements

- Python >= 3.9;
- [EasyBuild](https://easybuild.io/): a tool to build and install scientific software from source on High-Performance Computing (HPC) systems;
- [`llm`](https://pypi.org/project/llm) Python package: Python library for interacting with (remote and local) LLMs;
- [Ollama](https://github.com/ollama/ollama): tool to serve LLMs;
- [`llm-ollama`](https://pypi.org/project/llm-ollama) Python package: plugin for `llm` to access models running on an Ollama server;
    - particularly useful to use local LLMs;

### Installing Ollama

See [Ollama installation instructions](https://github.com/ollama/ollama/blob/main/README.md).

You should be able to run the `ollama` command in your shell environment, for example
```shell
$ ollama --version
ollama version is 0.9.0
```

### Creating Python virtual environment

```shell
python3 -m venv venv-eb-llm
source venv-eb-llm/bin/activate
```

#### Installing EasyBuild

**Note**: `easybuild-llm` requires recent changes to EasyBuild framework, including:
* [additional colors (`easybuild-framework` PR #4907)](https://github.com/easybuilders/easybuild-framework/pull/4907)
* [trigger post_run_shell_cmd_hook before raising error if shell command failed (`easybuild-framework` PR #4908)](https://github.com/easybuilders/easybuild-framework/pull/4908)

Until these changes are included in an EasyBuild release, you need to install the `develop` branches of EasyBuild:
```shell
pip install https://github.com/easybuilders/easybuild-framework/archive/develop.tar.gz
pip install https://github.com/easybuilders/easybuild-easyblocks/archive/develop.tar.gz
pip install https://github.com/easybuilders/easybuild-easyconfigs/archive/develop.tar.gz
# also install useful optional dependencies of EasyBuild
pip install archspec rich
```

#### Installing `llm` + `llm-ollama`

```shell
pip install llm llm-ollama
```

#### Checking environment

Check with `pip list` if all required packages are installed:

```shell
$ pip list
...
easybuild-easyblocks  5.1.1.dev0
easybuild-easyconfigs 5.1.1.dev0
easybuild-framework   5.1.1.dev0
...
llm                   0.26
llm-ollama            0.11
```

Check whether the `llm` command works, and whether the `llm-ollama` plugin is installed:

```shell
$ llm --version
llm, version 0.26

$ llm plugins
[
  {
    "name": "llm-ollama",
    "hooks": [
      "register_commands",
      "register_embedding_models",
      "register_models"
    ],
    "version": "0.11"
  }
]
```

Make sure that Ollama is running:

```
$ ollama --version
ollama version is 0.8.0
```

If you see a warning like "`could not connect to a running Ollama instance`", you may need to start Ollama first (in the background):

```
ollama serve &
```

## Setup & configuration

### Available LLMs

You need to make sure that one or more LLMs are available to be used on your system.

With `llm models`, you can check which LLMs you can use:

```
$ llm models
...
OpenAI Chat: o1
...
Ollama: gemma3:1b
...
```

Some of these models are remote (like the OpenAI ones), and an API key is required to use them (see also the [`llm` documentation](https://llm.datasette.io/en/stable/setup.html#api-key-management)).

### Local LLMs via Ollama

Thanks to Ollama, we can also easily download and use local LLMs.

With [`ollama list`](https://github.com/ollama/ollama/blob/main/README.md#list-models-on-your-computer) you can check with LLMs are installed on your system:

```
$ ollama list
NAME          ID              SIZE      MODIFIED
gemma3:1b     8648f39daa8f    815 MB    24 hours ago
```

To download addition models, you can use [`ollama pull`](https://github.com/ollama/ollama/blob/main/README.md#pull-a-model).

The list of models supported by Ollama is available [here](https://ollama.com/search).

### Configuring `easybuild-llm`

To specify which LLM should be used in EasyBuild, you must define the `$EB_LLM_MODEL` environment variable.

For example:
```
export EB_LLM_MODEL='gemma3:1b'
```

*Note: Gemma 3 1B is not a very capable model, but it's lightweight (815MB download), and hence makes for a good starting point.*


## Usage

By design, the functionality provided by `easybuild-llm` is included in EasyBuild itself, but is maintained in a totally separate Python packages.

To use LLMs in EasyBuild, you need to implement a couple of [EasyBuild hooks](https://docs.easybuild.io/hooks/):

* In the `start` hook, you should call the `init_llm_integration` provided in `easybuild.llm`:
  ```python
  from easybuild.llm import init_llm_integration

  def start_hook():
      init_llm_integration()
  ```
  This mainly checks whether all requirements are met (i.e. if `llm` is available, and if the LLM to use is specified via `$EB_LLM_MODEL`).

* To let an LLM explain why a shell command that was run by EasyBuild failed, you can use the `explain_failed_shell_cmd` and `format_llm_result` functions:
  ```python
  from easybuild.llm import explain_failed_shell_cmd, format_llm_result
  from easybuild.tools.output import print_error

  def post_run_shell_cmd_hook(cmd, **kwargs):
      shell_cmd_result = kwargs['shell_cmd_result']
      if shell_cmd_result.exit_code != 0:
          llm_result = explain_failed_shell_cmd(shell_cmd_result)
          print_error(format_llm_result(llm_result))
  ```

## Recommendations

### EasyBuild configuration

Since hook functions will be triggered regularly, you should configure EasyBuild to not print a message every time:

```shell
export EASYBUILD_SILENCE_HOOK_TRIGGER=1
```

Without doing this, you'll frequently see trace messages in the output produced by EasyBuild:
```
== Running post-run_shell_cmd hook...
```

### Model to use

To specify which LLM should be used, define the `$EB_LLM_MODEL` environment variable.

Picking an LLM to use is not easy, because there's a wide variety of choices available, and they are all a bit different in terms of capabilities, focus, "knowledge", etc.

One important aspect is the resources they require. To give an idea: a 7B variant of an LLM, which means it has 7 billion parameters, requires about 8GB of memory (see also [Ollama documentation](https://github.com/ollama/ollama/blob/main/README.md#model-library)).

In addition, the larger the model, the more compute resources is required to query the LLM.

**It is strongly recommended to only use LLMs if you have a decent GPU and sufficient memory available in your system.**

## Examples

### `make` not found (using Gemma3 1B as LLM)

Trivial example of output generation with installation that failed because the `make` command was not available, using [Gemma3 1B](https://ollama.com/library/gemma3:1b) as LLM.

```shell
$ export EB_LLM_MODEL='gemma3:1b'
$ eb --hooks eb-llm-hooks.py example.eb
...
Shell command 'make  -j 8' failed! (exit code 127)
Large Language Model 'gemma3:1b' explains it as follows:

> The error message "command not found" indicates that the shell cannot locate the
> `make` executable. This is because the shell is trying to execute a command,
> but it doesn't know where to find it.
>
> The issue is that the `make` executable is not in the system's `PATH`
> environment variable. This means the shell isn’t aware of the location where
> `make` is located.

*** NOTE: the text above was produced by an AI model, it may not be fully accurate! ***
(time spent querying LLM: 3.885 sec | tokens used: input=149, output=89)


ERROR: Shell command failed!
    full command              ->  mmake  -j 8
    exit code                 ->  127
    called from               ->  'build_step' function in /Users/example/easybuild-easyblocks/easybuild/easyblocks/generic/configuremake.py (line 382)
    working directory         ->  /tmp/build/example/1.2.3/system-system/example-1.2.3
    output (stdout + stderr)  ->  /tmp/eb-d3adb33f/run-shell-cmd-output/make-c0ff33/out.txt
    interactive shell script  ->  /tmp/eb-d3adb33f/run-shell-cmd-output/make-c0ff33/cmd.sh
```
