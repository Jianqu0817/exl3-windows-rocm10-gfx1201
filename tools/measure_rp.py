"""Time one public RP request; preserve the model's natural stopping policy."""
import argparse,json,os,sys,time,urllib.request
from pathlib import Path
from tokenizers import Tokenizer
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import server
ap=argparse.ArgumentParser()
ap.add_argument('--input-tokens',type=int,default=3872)
ap.add_argument('--case',type=int,choices=(1,2,3),default=1)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--prepare-only',action='store_true')
args=ap.parse_args()
target=args.input_tokens
if not 512<=target<=130000:ap.error('input-tokens must be 512..130000')
model=Path(os.environ.get('EXL3_MODEL_DIR',root.parents[1]/'models/Huihui-Qwen3.8-27B-abliterated-EXL3-3.0bpw'))
tpl=server.HFTemplate((model/'chat_template.jinja').read_text(encoding='utf8'),'', '<|im_end|>')
tok=Tokenizer.from_file(str(model/'tokenizer.json'))
places=['候船室','北堤','船厂','渡口','水文站','旧灯塔','柳湾小学']
things=['进水的手表','退回的船票','断了齿的钥匙','盖错章的信','被划掉的名字','刻着水位的砖']
notes=['小林说是昨天落下的，阿岑却记得，昨天镇上停了电。','窗台已经擦干净了，唯独角落里留着半圈水渍。','他们把两份记录摊开，有一行时间足足差了一刻钟。','值班人伸手想拿回去，看见门口有人，又把手缩了回来。']
system='你是中文长篇小说的续写者。请承接档案，用具体行动、对话和现场细节展开场景。'
ends=['夜班渡船本该停航，却突然响了两声汽笛。阿岑赶到码头，拦住准备解缆的船工。','灯塔守夜人打开铁柜，发现二十年前失踪的船长刚在交班簿上签过名。','小林走进等候室，一个陌生孩子把母亲的旧船票塞进他手里，却不肯说从哪里来。']
q=args.case-1;ending=ends[q]
tail='\n【现在的场景】'+ending+'请从此刻开始，连续写一章约两千汉字的中文角色扮演正文，展开调查、对质和选择的过程。不要列大纲，不要用总结代替故事。'
def text(n,pad):return '【河港档案】\n'+''.join(f'记录{i+1:04d}｜{places[i%7]}发现{things[(i//3)%6]}。{notes[i%4]}\n' for i in range(n))+(' x'*pad)+tail
def render(t):return tpl.render([{'role':'system','content':system},{'role':'user','content':t}],False,{},None)
def count(t):return len(tok.encode(render(t),add_special_tokens=False).ids)
lo,hi=0,target//16+1
while lo<hi:
   m=(lo+hi+1)//2
   if count(text(m,0))<=target:lo=m
   else:hi=m-1
pad=target-count(text(lo,0))
for _ in range(4):
   t=text(lo,pad);n=count(t)
   if n==target:break
   pad+=target-n
assert n==target,(n,pad)
body=dict(model='huihui-qwen3.8-27b',messages=[dict(role='system',content=sys),dict(role='user',content=t)],
          temperature=.8,min_p=.08,top_p=1,top_k=0,repetition_penalty=1,max_tokens=512,seed=1234+q,
          enable_thinking=False,ensure_sentence_end=True,stream=False)

if args.prepare_only:
 print('prepared prompt tokens:',n);sys.exit(0)
if args.out.exists():ap.error('Choose a new output file')
args.out.parent.mkdir(parents=True,exist_ok=True)
body.update(stream=True,stream_options={'include_usage':True},max_tokens=512)
url='http://127.0.0.1:18543'
def get(u):
 with urllib.request.urlopen(url+u,timeout=3) as r:return json.load(r)
deadline=time.monotonic()+240
while time.monotonic()<deadline:
 try:
  h=get('/health')
  if h.get('status')=='ok':break
 except OSError:pass
 time.sleep(2)
else:raise RuntimeError('not ready')
models=get('/v1/models');context=models['data'][0]['context_length']
if context<target+512+4:raise ValueError(f'Current context {context} cannot fit this prompt and reply')
if any(x['is_processing'] for x in get('/slots')):raise RuntimeError('A live chat already occupies the slot; use its server metrics instead.')
req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
samples=[(0,0.)];text=[];terminal=None;done=False
with urllib.request.urlopen(req,timeout=1800) as r:
 for raw in r:
  if not raw.startswith(b'data:'):continue
  payload=raw[5:].strip()
  if payload==b'[DONE]':done=True;break
  v=json.loads(payload)
  if 'error' in v:raise RuntimeError(v['error'])
  choice=v['choices'][0];text.append(choice['delta'].get('content') or '')
  if tim:=v.get('timings'):
   count,ms=tim['predicted_n'],tim['predicted_ms']
   if count>0:samples.append((count,ms/1000))
  if choice['finish_reason'] is not None:terminal=v
assert done and terminal and ''.join(text).strip()
m=terminal['exl3'];u=terminal['usage'];N=u['completion_tokens'];assert N==samples[-1][0] and u['prompt_tokens']==target
def bounds(n):
 if n==0 or n==N:return [0,0] if n==0 else [samples[-1][1]]*2
 # A timestamp at count n is an upper bound; the last observation below n is a lower bound.
 return [max(s for k,s in samples if k<n),min(s for k,s in samples if k>=n)]
ranges=[]
for end in range(128,N+1,128):
 left,right=bounds(end-128),bounds(end)
 ranges.append([128/(right[1]-left[0]),128/(right[0]-left[1]) if right[0]>left[1] else None])
result=dict(id=terminal['id'],context=context,usage=u,metrics=m,finish=terminal['choices'][0]['finish_reason'],windows128_rate_bounds=ranges,samples=samples)
args.out.write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='samples'},ensure_ascii=False))
