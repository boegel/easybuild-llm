##
# Copyright 2025-2025 Ghent University
#
# This file is part of EasyBuild,
# originally created by the HPC team of Ghent University (http://ugent.be/hpc/en),
# with support of Ghent University (http://ugent.be/hpc),
# the Flemish Supercomputer Centre (VSC) (https://www.vscentrum.be),
# Flemish Research Foundation (FWO) (http://www.fwo.be/en)
# and the Department of Economy, Science and Innovation (EWI) (http://www.ewi-vlaanderen.be/en).
#
# https://github.com/easybuilders/easybuild-llm
#
# EasyBuild is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation v2.
#
# EasyBuild is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EasyBuild.  If not, see <http://www.gnu.org/licenses/>.
##
"""
Integration of LLMs in EasyBuild.

Authors:
- Kenneth Hoste (Ghent University)
"""

import importlib.metadata
import os
import textwrap

# allow importing of easybuild.llm without actually having the 3rd party 'llm' Python pacakge available
try:
    import llm
except ImportError:
    pass

from collections import namedtuple

from easybuild.base import fancylogger
from easybuild.tools.build_log import EasyBuildError
from easybuild.tools.output import COLOR_CYAN, colorize


EXPLAIN_FAILED_SHELL_CMD_PROMPT = """
%(output)s

Explain why the '%(cmd)s' shell command failed with the above output.
The shell command was running in %(work_dir)s, and had %(exit_code)s as exit code.

Start with pointing out the actual error message from the output.
Then explain what that error means, and what caused it.
Do not make suggestions on how to fix the problem, only explain.
Keep it short and to the point.
"""

LLM_ACTION_EXPLAIN_FAILED_SHELL_CMD = 'explain-failed-shell-cmd'
LLM_ACTIONS = [
    LLM_ACTION_EXPLAIN_FAILED_SHELL_CMD,
]


_log = fancylogger.getLogger('llm', fname=False)


LLMConfig = namedtuple('LLMConfig', ('model_name',))
LLMResult = namedtuple('LLMResult', ('model_name', 'info', 'answer', 'duration_secs', 'input_tokens', 'output_tokens'))


def get_model(model_name=None):
    """
    Get instance of LLM model we can query
    """
    model_name = os.getenv('EB_LLM_MODEL')

    # on LLM model to use *must* be specified, and it must be a known model (to 'llm' Python package)
    if model_name:
        try:
            model = llm.get_model(model_name)
            return model
        except llm.UnknownModelError:
            raise EasyBuildError(f"Unknown LLM model specified: {model_name}")
    else:
        raise EasyBuildError("LLM model to use is not specified" + common_err_suffix_req)


def init_llm_integration():
    """
    Initialise integration with LLMs:
    - verify whether 'llm' Python package is available;
    - verify configuration settings for LLM integration;
    """
    common_err_suffix_req = ", this is required when integration with LLMs is enabled!"

    try:
        llm_version = importlib.metadata.version('llm')
    except importlib.metadata.PackageNotFoundError:
        raise EasyBuildError("'llm' Python package is not available" + common_err_suffix_req)
    _log.info(f"Found version {llm_version} of 'llm' Python package")

    model = get_model()

    return LLMConfig(model_name=model.model_id)


def explain_failed_shell_cmd(shell_cmd_res):
    """
    Query LLM to explain failed shell command
    """

    prompt = EXPLAIN_FAILED_SHELL_CMD_PROMPT % {
        'cmd': shell_cmd_res.cmd,
        'exit_code': shell_cmd_res.exit_code,
        'output': shell_cmd_res.output,
        'work_dir': shell_cmd_res.work_dir,
    }

    model = get_model()
    model_name = model.model_id

    _log.info(f"Querying LLM '{model_name}' using following prompt: {prompt}")
    response = model.prompt(prompt)
    explanation = response.text().lstrip()
    _log.info(f"Result from querying LLM: {explanation}")

    lines = explanation.splitlines()
    answer = []
    for line in lines:
        if line:
            answer.extend(textwrap.wrap(line, width=80, replace_whitespace=False)) # + [''])
        else:
            answer.append('')
    answer = '\n'.join(answer)

    duration_secs = response.duration_ms() / 1000

    token_usage = response.usage()
    input_tokens, output_tokens = token_usage.input, token_usage.output

    info = f"Shell command '{shell_cmd_res.cmd}' failed! (exit code {shell_cmd_res.exit_code})"
    return LLMResult(model_name=model_name, info=info, answer=answer, duration_secs=duration_secs,
                     input_tokens=input_tokens, output_tokens=output_tokens)


def format_llm_result(llm_result):
    """
    Format LLM result for printing
    """
    lines = [
        '',
        llm_result.info,
        f"Large Language Model '{llm_result.model_name}' explains it as follows:",
        '',
    ]
    lines.extend('> ' + x for x in llm_result.answer.split('\n'))
    lines.extend([
        '',
        "*** NOTE: the text above was produced by an AI model, it may not be fully accurate! ***",
        f"(time spent querying LLM: {llm_result.duration_secs} sec "
        f"| tokens used: input={llm_result.input_tokens}, output={llm_result.output_tokens})",
    ])
    return colorize('\n'.join(lines), COLOR_CYAN)
