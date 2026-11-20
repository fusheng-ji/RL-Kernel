import re,sys,os,json
OUT=sys.argv[1]
def parse(d):
    r={}
    rc={}
    for l in open(f"{d}/rc.txt"):
        f,c=l.rsplit(" rc=",1); rc[f]=int(c)
    for f,c in rc.items():
        lg=f"{d}/{f.replace('/','_')}.log"
        t=open(lg,errors='replace').read() if os.path.exists(lg) else ""
        last=[x for x in t.strip().splitlines() if re.search(r"\b(passed|failed|skipped|error|errors|no tests ran)\b",x) and ("in " in x)]
        s=last[-1] if last else ""
        cnt={k:0 for k in("passed","failed","skipped","error")}
        for n,k in re.findall(r"(\d+) (passed|failed|skipped|errors?)",s):
            cnt["error" if k.startswith("error") else k]+=int(n)
        r[f]=(cnt,c)
    return r
res={}
rows=[]
for A in (0,1):
    h=parse(f"{OUT}/head/ops-autotune{A}"); m=parse(f"{OUT}/main/ops-autotune{A}")
    res[A]=(h,m)
md=[]
for A in (0,1):
    h,m=res[A]; md.append(f"\n### RL_KERNEL_AUTOTUNE={A}\n")
    md.append("| file | head P/F/S/E (rc) | main P/F/S/E (rc) | verdict |\n|---|---|---|---|")
    for f in sorted(set(h)|set(m)):
        if f in h and f in m:
            a,b=h[f],m[f]
            fa=fmt=lambda x:f"{x[0]['passed']}/{x[0]['failed']}/{x[0]['skipped']}/{x[0]['error']} ({x[1]})"
            if a==b: v="same"
            elif (a[0]['failed']+a[0]['error']>b[0]['failed']+b[0]['error']) or (a[1]!=0 and b[1]==0): v="**head only** (fail)"
            elif (b[0]['failed']+b[0]['error']>a[0]['failed']+a[0]['error']) or (b[1]!=0 and a[1]==0): v="**main only** (fail)"
            else: v="differs (counts only)"
            md.append(f"| `{f}` | {fmt(a)} | {fmt(b)} | {v} |")
    only_h=[f for f in h if f not in m]; only_m=[f for f in m if f not in h]
    md.append(f"\nFiles only on head: {only_h}\n\nFiles only on main: {only_m}\n")
open(f"{OUT}/ops-table.md","w").write("\n".join(md)); print("\n".join(md))
