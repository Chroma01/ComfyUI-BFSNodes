"""Render an explanatory GIF from the published temporal penalty, CPU only."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import math
R=Path(__file__).resolve().parent
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):return ImageFont.truetype(FONT,n)
W,H=1000,590
frames=[]
names=['Free window w','Falloff f','Strength lambda','Cap c']
descriptions=[('Time tolerance around the matching moment.','Wider window: more nearby times receive no penalty.'),('How gradually preference fades outside that window.','Smaller falloff: the preference drops more sharply.'),('How strongly distant reference times are penalized.','This is temporal bias strength, not LoRA strength.'),('The maximum penalty for distant reference times.','The cap limits suppression; it does not delete frames.')]
for phase in range(4):
 for j in range(48):
  q=(1-math.cos(2*math.pi*j/48))/2
  vals=[.2,.25,2.,6.]
  vals[phase]=[(.05+.4*q),(.1+.5*q),(.5+3.5*q),(1.+7*q)][phase]
  w,f,lam,c=vals
  im=Image.new('RGB',(W,H),'#0c1320');d=ImageDraw.Draw(im)
  d.text((28,16),'TARA: which moment should the output consult?',font=font(26),fill='white')
  d.text((28,58),f'{phase+1}/4  '+names[phase],font=font(23),fill='#a9dbff')
  d.text((28,94),descriptions[phase][0],font=font(18),fill='#d1deef')
  d.text((28,120),descriptions[phase][1],font=font(18),fill='#9cb0ca')
  for k,(label,v) in enumerate(zip(['w','f','lambda','c'],vals)):
   x=28+k*242;d.rounded_rectangle((x,160,x+226,204),8,fill='#253c4d' if phase==k else '#152334')
   d.text((x+12,171),f'{label} = {v:.2f}'+(' s' if k<2 else ''),font=font(19),fill='#7cf0c9' if phase==k else '#d1deef')
  x0,x1=65,935
  def xx(t):return x0+(t-4)/2*(x1-x0)
  d.text((28,222),'Output is generating the moment at 5.00 s',font=font(19),fill='#ffc986')
  d.text((28,247),'Reference timeline: brighter means less suppression',font=font(16),fill='#9cb0ca')
  d.rectangle((xx(5)-2,270,xx(5)+2,422),fill='#ffc986')
  d.rounded_rectangle((xx(5-w),270,xx(5+w),320),8,fill='#153c36')
  d.line((x0,295,x1,295),fill='#60728a',width=2)
  for i in range(41):
   t=4+i*.05;p=min(c,lam*(max(0,abs(t-5)-w)/f)**2);weight=math.exp(-p)
   color=tuple(round(a+(b-a)*weight) for a,b in zip((21,31,45),(105,245,198)))
   x=xx(t);d.ellipse((x-6,289,x+6,301),fill=color,outline='#64748b')
  for t in [4,4.5,5,5.5,6]:d.text((xx(t)-20,330),f'{t:.1f}s',font=font(15),fill='#b8c7db')
  d.text((28,370),'Relative reference weight',font=font(17),fill='#a8bad1')
  d.line((x0,420,x0,515,x1,515),fill='#536781',width=1)
  pts=[]
  for i in range(401):
   t=4+2*i/400;weight=math.exp(-min(c,lam*(max(0,abs(t-5)-w)/f)**2))
   pts.append((xx(t),515-weight*95))
  d.line(pts,fill='#69f5c6',width=3)
  d.text((25,412),'1',font=font(13),fill='#9fb1c8');d.text((25,505),'0',font=font(13),fill='#9fb1c8')
  d.text((28,548),'Illustrative times, not RGB-frame sampling. Relative weighting, not final attention probability.',font=font(16),fill='#99aec8')
  frames.append(im)
frames[10].save(R/'tara_parameters_poster.png')
palette=frames[24].quantize(colors=128)
images=[f.quantize(palette=palette,dither=Image.Dither.NONE) for f in frames]
images[0].save(R/'tara_parameters.gif',save_all=True,append_images=images[1:],duration=125,loop=0,optimize=True,disposal=2)
print(R/'tara_parameters.gif')
