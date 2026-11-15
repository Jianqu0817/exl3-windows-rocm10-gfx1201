"""CPU reproduction using the deployed model template, without loading model weights."""
import argparse
import copy
import importlib.util
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('http_adapter', ROOT / 'server.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
source = (Path(os.environ.get('EXL3_MODEL_DIR', ROOT.parents[1] / 'models/Huihui-Qwen3.8-27B-abliterated-EXL3-3.0bpw')) / 'chat_template.jinja').read_text(encoding='utf-8')
template = adapter.HFTemplate(source, '', '<|im_end|>')


def check(messages):
    original = copy.deepcopy(messages)
    normalized = adapter.template_messages(messages)
    assert messages == original
    systems = [adapter.text_content(m.get('content')) for m in messages if m['role'] == 'system']
    if systems:
        assert normalized[0] == {'role': 'system', 'content': '\n\n'.join(systems)}
        assert sum(m['role'] == 'system' for m in normalized) == 1
    prompt = template.render(normalized, False)
    assert prompt.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests-dir', type=Path)
    args = parser.parse_args()
    plain = [{'role': 'system', 'content': 'SINGLE'}, {'role': 'user', 'content': 'QUESTION'}]
    assert check(plain) == template.render(plain, False)
    combined = [
        {'role': 'system', 'content': 'ROLE_CARD'},
        {'role': 'system', 'content': [{'type': 'text', 'text': 'WORLD_INFO'}]},
        {'role': 'assistant', 'content': 'GREETING'},
        {'role': 'system', 'content': 'DEPTH_NOTE'},
        {'role': 'user', 'content': 'LATEST_QUESTION'},
        {'role': 'system', 'content': 'POST_HISTORY'},
    ]
    prompt = check(combined)
    for text in ('ROLE_CARD', 'WORLD_INFO', 'GREETING', 'DEPTH_NOTE', 'LATEST_QUESTION', 'POST_HISTORY'):
        assert prompt.count(text) == 1
    assert prompt.index('GREETING') < prompt.index('LATEST_QUESTION')
    tool = [*plain, {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"query":"x"}'}}]},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'RESULT'},
        {'role': 'system', 'content': 'AFTER_TOOL_NOTE'}]
    prompt = check(tool)
    assert '<tool_call>' in prompt and '<tool_response>' in prompt and 'RESULT' in prompt
    prefix = '角色：未完的一句  '
    continuation = [*plain, {'role': 'assistant', 'content': prefix}]
    normalized = adapter.template_messages(continuation)
    prompt = template.render(normalized, False)
    assert prompt.endswith(prefix) and prompt.count('<|im_start|>assistant') == 1
    assert '__EXL3_CONTINUE_' not in prompt
    assert continuation[-1]['content'] == prefix
    for think in (False, True):
        assert template.render(adapter.template_messages([*plain, {'role': 'assistant', 'content': ''}]), think) == template.render(plain, think)
    checked = []
    if args.requests_dir:
        for path in sorted(args.requests_dir.glob('llm-api-*.request.json')):
            body = json.loads(path.read_text(encoding='utf-8'))
            check(body['messages'])
            checked.append(path.name)
    print(json.dumps({'status': 'PASS', 'synthetic_cases': 6, 'actual_request_files': checked}))


if __name__ == '__main__':
    main()
