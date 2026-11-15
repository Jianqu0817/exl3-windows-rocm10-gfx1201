"""Small live contract check: models, JSON reply, SSE reply and prefix reuse."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-url',default='http://127.0.0.1:18543/v1')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    def get(path):
        with urlopen(args.base_url+path,timeout=30) as response:
            return json.load(response)
    models=get('/models')
    model=models['data'][0]['id']
    assert model=='huihui-qwen3.8-27b',model
    body=dict(model=model,messages=[{'role':'system','content':'简短回答，最多两句话。'},
        {'role':'user','content':('背景资料：这里是春天的图书馆，窗边有一只猫。\n'*100)+'请用中文描述窗边的猫。'}],
        max_tokens=32,temperature=.8,top_p=1,top_k=0,min_p=.08,repetition_penalty=1,
        presence_penalty=0,frequency_penalty=0,enable_thinking=False,n=1)
    def request(stream):
        payload={**body,'stream':stream}
        if stream:
            payload['stream_options']={'include_usage':True}
        return Request(args.base_url+'/chat/completions',data=json.dumps(payload).encode(),
                       headers={'Content-Type':'application/json'},method='POST')
    with urlopen(request(False),timeout=180) as response:
        complete=json.load(response)
    assert complete['choices'][0]['message']['content'].strip()
    assert not complete['choices'][0]['message']['reasoning_content']
    assert 'cached_tokens' in complete['usage']['prompt_tokens_details']
    chunks=[];done=False
    with urlopen(request(True),timeout=180) as response:
        assert response.headers['Content-Type'].startswith('text/event-stream')
        for raw in response:
            line=raw.decode('utf-8').strip()
            if not line.startswith('data: '):
                continue
            if line=='data: [DONE]':
                done=True;break
            chunks.append(json.loads(line[6:]))
    text=''.join(x['choices'][0]['delta'].get('content','') for x in chunks)
    assert done and text.strip()
    assert not any(x['choices'][0]['delta'].get('reasoning_content') for x in chunks)
    final=next(x for x in reversed(chunks) if x.get('usage'))
    assert final['usage']['prompt_tokens_details']['cached_tokens']>0
    assert final['choices'][0]['finish_reason'] in ('stop','length')
    result={'status':'PASS','model':model,'base_url':args.base_url,
            'nonstream_usage':complete['usage'],'stream_usage':final['usage'],
            'stream_chunks':len(chunks),'nonstream_text':complete['choices'][0]['message']['content'],
            'stream_text':text,'done':done,'thinking':False}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
