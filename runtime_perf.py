import torch, vllm_xpu_kernels._C
def t(fn,n=20):
    for _ in range(3): fn()
    torch.xpu.synchronize(); s=torch.xpu.Event(enable_timing=True); e=torch.xpu.Event(enable_timing=True)
    s.record(); [fn() for _ in range(n)]; e.record(); torch.xpu.synchronize(); return s.elapsed_time(e)/n
rows,start,k=2560,99840,512; cols=(start+rows)//2
ends=torch.clamp((start+torch.arange(rows,device='xpu',dtype=torch.int32)+1)//2,min=1).to(torch.int32)
x=torch.randn(rows,cols,device='xpu'); st=torch.zeros(rows,device='xpu',dtype=torch.int32); out=torch.empty(rows,k,device='xpu',dtype=torch.int32)
p=t(lambda: torch.ops._C.top_k_per_row_prefill(x,st,ends,out,rows,x.stride(0),x.stride(1),k))
y=torch.randn(256,51200,device='xpu'); sl=torch.full((256,1),51200,device='xpu',dtype=torch.int32); o2=torch.empty(256,k,device='xpu',dtype=torch.int32)
d=t(lambda: torch.ops._C.top_k_per_row_decode(y,1,sl,o2,256,y.stride(0),y.stride(1),k))
print(f"RESULT prefill {p*1000:.1f}us  decode {d*1000:.1f}us")
