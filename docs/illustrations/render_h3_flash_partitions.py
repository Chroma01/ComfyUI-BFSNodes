"""Render explanatory figures; no model inference or GPU required."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

OUT=Path(__file__).parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'savefig.facecolor':'white'})

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)

# Conceptual groups: equal spacing within each video, not actual H3 token counts.
fig,ax=plt.subplots(figsize=(10,7))
n=170;rgb=np.full((n,n,3),.91)
t=np.linspace(0,4,60);dt=np.abs(t[:,None]-t[None,:])
p=np.minimum(2*(np.maximum(dt-.2,0)/.25)**2,6);weights=np.exp(-p)
rgb[110:170,40:100]=plt.get_cmap('YlOrRd')(weights)[...,:3]
ax.imshow(rgb,interpolation='nearest',origin='upper')
bounds=[0,20,40,100,110,170];names=['Text','Pictures','Motion video','Audio','Output video']
centers=[(a+b-1)/2 for a,b in zip(bounds,bounds[1:])]
ax.set_xticks(centers,names);ax.set_yticks(centers,names)
ax.xaxis.tick_top();ax.xaxis.set_label_position('top')
ax.set_xlabel('Keys: what the model can consult',labelpad=14)
ax.set_ylabel('Queries: what is being updated')
for b in bounds[1:-1]:ax.axhline(b-.5,color='white',lw=2);ax.axvline(b-.5,color='white',lw=2)
ax.add_patch(Rectangle((39.5,109.5),60,60,fill=False,edgecolor='#111827',lw=2))
ax.set_title('The temporal bias changes one family of attention edges',pad=55,weight='bold')
fig.text(.12,.02,'Gray: unchanged score rule. Colored block: output queries consulting the motion video.\nNear-time keys retain weight 1; distant keys retain a small positive weight. No hard mask.',fontsize=10)
sm=plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='YlOrRd');fig.colorbar(sm,ax=ax,shrink=.65,label='Unnormalized weight multiplier exp(-penalty)')
save(fig,'h3_temporal_attention_scope')

fig,ax=plt.subplots(figsize=(11,5))
x=np.linspace(0,1.1,600);penalty=np.minimum(2*(np.maximum(x-.2,0)/.25)**2,6)
baseline=np.exp(-6);full=np.exp(-penalty);correction=full-baseline
ax.plot(x,full,color='#172554',lw=3,label='Required weight: exp(-penalty)')
ax.plot(x,np.full_like(x,baseline),color='#ea580c',lw=2,ls='--',label='Global guide contribution: exp(-6)')
ax.fill_between(x,baseline,full,color='#22c55e',alpha=.35,label='Positive local correction: exp(-penalty) - exp(-6)')
ax.axvline(.2,color='#64748b',ls=':');ax.axvline(.2+.25*np.sqrt(3),color='#64748b',ls=':')
ax.text(.205,.91,'0.20 s\nfree window',fontsize=10)
ax.text(.65,.5,'Beyond ~0.633 s:\ncorrection is zero,\nbaseline remains',fontsize=10)
ax.set(xlabel='Absolute time difference between output and guide (seconds)',ylabel='Multiplier of unnormalized softmax weight',ylim=(-.03,1.08),xlim=(0,1.1))
ax.set_title('Same temporal rule, decomposed into positive contributions',weight='bold')
ax.legend(loc='upper right',fontsize=9);ax.grid(alpha=.18)
fig.tight_layout();save(fig,'h3_temporal_weight_decomposition')

fig,ax=plt.subplots(figsize=(14,7));ax.set(xlim=(0,14),ylim=(0,7));ax.axis('off')
def box(x,y,w,h,text,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',facecolor=color,edgecolor='#475569',lw=1.2))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.6,color='#475569'))
box(.2,2.8,2,1.2,'Output-video\nquery chunk','#dbeafe')
parts=[(5.05,'Other keys\nweight = 1','#ede9fe'),(2.85,'All motion-video keys\nweight = exp(-cap)','#ffedd5'),(.65,'Nearby motion-frame keys\nweight = exp(-penalty) - exp(-cap)','#dcfce7')]
for y,text,color in parts:
 box(3,y,4,1.3,text,color);arrow((2.3,3.4),(2.8,y+.65))
 box(7.7,y,2.3,1.3,'Native FlashAttention\noutput + logsumexp',color);arrow((7.15,y+.65),(7.5,y+.65));arrow((10.15,y+.65),(10.8,3.4))
box(11,2.65,2.7,1.5,'Stable weighted merge\n\nSame biased softmax','#dbeafe')
ax.text(7,6.7,'Flash Partitions: reuse dense kernels instead of compiling a per-score rule',ha='center',weight='bold',fontsize=13)
ax.text(.2,.04,'Guide contributions overlap intentionally; their weights sum to the original weight.\nQueries outside the output video use ordinary attention. Spatial tokens are retained; corrections are batched by frame.',fontsize=10)
save(fig,'h3_flash_partitions_flow')
print('Rendered 3 PNG and 3 SVG figures')
