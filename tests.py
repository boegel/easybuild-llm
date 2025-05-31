import os
import pytest
from collections import namedtuple
from unittest.mock import Mock, patch

from easybuild.tools.build_log import EasyBuildError
from easybuild.tools.run import RunShellCmdResult
from easybuild.llm import explain_failed_shell_cmd, format_llm_result, get_model, init_llm_integration


class FakeUnknownModelError(Exception):
    pass


@patch('easybuild.llm.llm')
def test_get_model(mocked_llm):
    """
    Test get_model function
    """

    mocked_model = Mock()
    mocked_model.model_id = 'dummy-llm' 
    mocked_llm.UnknownModelError = FakeUnknownModelError

    def mocked_get_model(name):
        if name == 'dummy-llm':
            return mocked_model
        else:
            raise FakeUnknownModelError("no such model")

    mocked_llm.get_model = mocked_get_model

    # $EB_LLM_MODEL not defined means trouble
    with pytest.raises(EasyBuildError) as excinfo:
        get_model()
    assert "LLM model to use is not specified, must be specified via $EB_LLM_MODEL" in excinfo.value.msg

    # unknown model specified
    os.environ['EB_LLM_MODEL'] = 'no-such-model'
    with pytest.raises(EasyBuildError) as excinfo:
        get_model()
    assert "Unknown LLM model specified: no-such-model" in excinfo.value.msg

    # known model, all good
    os.environ['EB_LLM_MODEL'] = 'dummy-llm'
    model = get_model()
    assert model.model_id == 'dummy-llm'


@patch('easybuild.llm.llm')
def test_init_llm_integration(mocked_llm):
    """
    Test init_llm_integration function
    """

    mocked_model = Mock()
    mocked_model.model_id = 'dummy-llm'
    mocked_llm.get_model.return_value = mocked_model

    llm_config = init_llm_integration()
    assert llm_config.model_name == 'dummy-llm'


@patch('easybuild.llm.llm')
@patch('easybuild.llm.colorize')
def test_explain_failed_shell_cmd_format_llm_result(mocked_colorize, mocked_llm):
    """
    Test both explain_failed_shell_cmd + format_llm_result functions
    """

    mocked_response = Mock()
    mocked_response.text = lambda: "\nResistance is futile,\nyou will be assimilated\n\nI'll be back"
    mocked_response.duration_ms = lambda: 123
    ResponseUsage = namedtuple('ResponseUsage', ('input', 'output'))
    mocked_response.usage = lambda: ResponseUsage(input=123, output=456)

    mocked_model = Mock()
    mocked_model.model_id = 'dummy-llm'
    mocked_model.prompt = lambda _: mocked_response
    mocked_llm.get_model.return_value = mocked_model

    shell_cmd_result = RunShellCmdResult(cmd='echo hello', exit_code=1, output='hello', stderr=None, work_dir='/tmp',
                                         out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)
    res = explain_failed_shell_cmd(shell_cmd_result)

    assert res.model_name == 'dummy-llm'
    assert res.info == "Shell command 'echo hello' failed! (exit code 1)"
    assert res.answer == "Resistance is futile,\nyou will be assimilated\n\nI'll be back"
    assert res.duration_secs == 0.123
    assert res.input_tokens == 123
    assert res.output_tokens == 456

    mocked_colorize.side_effect = lambda x, _: '<color>' + x + '</color>'

    formatted_res = format_llm_result(res)
    expected_formatted_res = '\n'.join([
        "<color>",
        "Shell command 'echo hello' failed! (exit code 1)",
        "Large Language Model 'dummy-llm' explains it as follows:",
        '',
        "> Resistance is futile,",
        "> you will be assimilated",
        "> ",
        "> I'll be back",
        '',
        "*** NOTE: the text above was produced by an AI model, it may not be fully accurate! ***",
        "(time spent querying LLM: 0.123 sec | tokens used: input=123, output=456)</color>",
    ])
    assert formatted_res == expected_formatted_res
