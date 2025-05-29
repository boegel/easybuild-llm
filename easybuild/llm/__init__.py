import importlib.metadata
import textwrap

# allow importing of easybuild.llm without actually having the 3rd party 'llm' Python pacakge available
try:
    import llm
except ImportError:
    pass

from collections import namedtuple
from datetime import datetime

from easybuild.base import fancylogger
from easybuild.tools.build_log import EasyBuildError, time_str_since
from easybuild.tools.config import build_option


EXPLAIN_FAILED_SHELL_COMMAND_PROMPT = """
%(output)s

Explain why the '%(cmd)s' shell command failed with the above output.
The shell command was running in %(work_dir)s, and had %(exit_code)s as exit code.

Start with pointing out the actual error message from the output.
Then explain what that error means, and what caused it.
Do not make suggestions on how to fix the problem, only explain.
Keep it short and to the point.
"""

LLM_ACTION_EXPLAIN = 'explain'
LLM_ACTIONS = [LLM_ACTION_EXPLAIN]


_log = fancylogger.getLogger('llm', fname=False)


LLMResult = namedtuple('LLMResult', ('model', 'answer', 'time_spent'))


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

    # on LLM model to use *must* be specified, and it must be a known model (to 'llm' Python package)
    llm_model = build_option('llm_model')
    if llm_model:
        try:
            model = llm.get_model(llm_model)
        except llm.UnknownModelError:
            raise EasyBuildError(f"Unknown LLM model specified: {llm_model}")
    else:
        raise EasyBuildError("LLM model to use is not specified" + common_err_suffix_req)

    # specified LLM actions must be known actions, and at least one must be specified
    llm_actions = build_option('llm_actions')
    known_llm_actions = "known LLMs actions: " + ', '.join(LLM_ACTIONS)
    unknown_llm_actions = [x for x in llm_actions or [] if x not in LLM_ACTIONS]
    if unknown_llm_actions:
        error_msg = "Unknown LLM action(s) specified: " + ', '.join(unknown_llm_actions) + f" ({known_llm_actions})"
        raise EasyBuildError(error_msg)

    if not llm_actions:
        raise EasyBuildError("No LLM actions specified" + common_err_suffix_req + f" ({known_llm_actions})")


def explain_failed_shell_cmd(shell_cmd_res):

    start_time = datetime.now()

    prompt = EXPLAIN_FAILED_SHELL_COMMAND_PROMPT % {
        'cmd': shell_cmd_res.cmd,
        'exit_code': shell_cmd_res.exit_code,
        'output': shell_cmd_res.output,
        'work_dir': shell_cmd_res.work_dir,
    }

    llm_model = build_option('llm_model')
    model = llm.get_model(llm_model)

    _log.info(f"Querying LLM {llm_model} using following prompt: {prompt}")
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

    time_spent = time_str_since(start_time)

    return LLMResult(model=llm_model, answer=answer, time_spent=time_spent)
