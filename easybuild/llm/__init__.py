import textwrap

# allow importing of easybuild.llm without actually having the 3rd party 'llm' Python pacakge available
try:
    import llm
except ImportError:
    pass

from collections import namedtuple
from datetime import datetime

from easybuild.base import fancylogger
from easybuild.tools.build_log import time_str_since
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


_log = fancylogger.getLogger('llm', fname=False)


LLMResult = namedtuple('LLMResult', ('answer', 'time_spent'))


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

    return LLMResult(answer=answer, time_spent=time_spent)
