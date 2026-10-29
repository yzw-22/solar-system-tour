#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳系巡航 · Solar System Grand Tour  (程序化逐帧渲染)
飞行器视角：太阳→水星→金星→地球/月球→火星→木星→土星→天王星→
海王星→冥王星/卡戎→赛德娜→望向比邻星与星空。
物理设定：太阳为唯一点光源（相位自动正确）；天体尺寸真实比例（太阳例外，已声明）；
轨道距离幂律压缩；星空程序化生成（银河带经过半人马座方向，与事实一致）。
"""
import numpy as np, os, sys, math, time, wave, subprocess, shutil
from PIL import Image, ImageDraw, ImageFont, ImageFilter

BASE = os.path.dirname(os.path.abspath(__file__))
TEX  = os.path.join(BASE, 'textures')
OUTF = os.path.join(BASE, 'frames')
PREV = os.path.join(BASE, 'preview')
VID  = os.path.join(BASE, 'video')
SKYCACHE = os.path.join(BASE, 'sky.npz')

def _find_ffmpeg():
    exe = os.environ.get('FFMPEG') or shutil.which('ffmpeg')
    if not exe:
        try:
            import imageio_ffmpeg; exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            raise SystemExit('未找到 ffmpeg：请将其加入 PATH，或 pip install imageio-ffmpeg')
    return exe
FFMPEG = _find_ffmpeg()

W, H = 1920, 1080
FPS  = 24
DUR  = 96.0          # 统一时间轴：每颗行星四段式（接近-特写-停留-离开）
DUR_OUT = DUR
NFRAMES = int(DUR_OUT*FPS)  # 2304

def old_time(tn):
    return tn
SEED = 20261024

# ---------------------------------------------------------------- 基础工具
def normalize(v):
    v = np.asarray(v, np.float64); return v/(np.linalg.norm(v)+1e-12)

def rot_x(a):
    c,s = math.cos(a), math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_z(a):
    c,s = math.cos(a), math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def catmull_rom(ts, pts, t):
    ts = np.asarray(ts,float); pts = np.asarray(pts,float)
    if t <= ts[0]: return pts[0].copy()
    if t >= ts[-1]: return pts[-1].copy()
    i = int(np.searchsorted(ts,t)-1); i = max(0,min(len(ts)-2,i))
    t0,t1,t2,t3 = ts[max(i-1,0)],ts[i],ts[i+1],ts[min(i+2,len(ts)-1)]
    p0,p1,p2,p3 = pts[max(i-1,0)],pts[i],pts[i+1],pts[min(i+2,len(pts)-1)]
    u = (t-t1)/(t2-t1); u2,u3 = u*u,u*u*u
    return 0.5*((2*p1)+(-p0+p2)*u+(2*p0-5*p1+4*p2-p3)*u2+(-p0+3*p1-3*p2+p3)*u3)

def cr_scalar(ts, vals, t):
    return catmull_rom(ts, np.asarray(vals,float)[:,None], t)[0]

_HERM_M = {}
def hermite_limited(ts, pts, t, kcap=2.5):
    """切线限幅 Hermite：每键切线按"相邻两段弦速度较小者"一次性限幅（C1 连续，速度无帧间抖动）"""
    ts = np.asarray(ts,float); pts = np.asarray(pts,float); n = len(ts)
    ck = (id(ts),id(pts))
    if ck not in _HERM_M:
        m = np.zeros_like(pts)
        for kk in range(n):
            a = pts[max(kk-1,0)]; b = pts[min(kk+1,n-1)]
            m[kk] = (b-a)/max(ts[min(kk+1,n-1)]-ts[max(kk-1,0)],1e-9)
        for kk in range(n):
            v_in = np.linalg.norm(pts[kk]-pts[max(kk-1,0)])/max(ts[kk]-ts[max(kk-1,0)],1e-9)
            v_out = np.linalg.norm(pts[min(kk+1,n-1)]-pts[kk])/max(ts[min(kk+1,n-1)]-ts[kk],1e-9)
            cap = kcap*min(v_in,v_out)
            nrm = float(np.linalg.norm(m[kk]))
            if nrm > cap: m[kk] *= cap/nrm
        _HERM_M[ck] = m
    m = _HERM_M[ck]
    i = int(np.clip(np.searchsorted(ts,t)-1,0,n-2))
    h = max(ts[i+1]-ts[i],1e-9)
    u = float(np.clip((t-ts[i])/h,0,1))
    h00=(1+2*u)*(1-u)**2; h10=u*(1-u)**2; h01=u*u*(3-2*u); h11=u*u*(u-1)
    return h00*pts[i]+h10*h*m[i]+h01*pts[i+1]+h11*h*m[i+1]

# ---------------------------------------------------------------- 噪声
def fbm_2d(w, h, seed, octaves=(3,6,12,24,48), gain=0.55):
    """FFT 周期噪声：x 方向可无缝环绕（适合经度贴图）。octaves=整幅图像的周期数。"""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:,None]; fx = np.fft.rfftfreq(w)[None,:]
    fr = np.sqrt(fx*fx+fy*fy)
    dm = min(w,h)
    acc = np.zeros((h,w), np.float64); amp=1.0; tot=0.0
    for k in octaves:
        spec = np.exp(1j*rng.uniform(0,2*math.pi,(h,w//2+1)))
        spec *= (np.abs(fr-k/dm) < k/dm*0.6+0.5/dm)
        f = np.fft.irfft2(spec, s=(h,w)); mx = np.abs(f).max()
        if mx < 1e-12: continue
        acc += amp*(f/mx); tot += amp; amp *= gain
    acc /= max(tot,1e-9)
    return ((acc-acc.min())/(acc.ptp()+1e-9)).astype(np.float32)

def fbm_1d(n, seed, k):
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0,2*math.pi,k); am = rng.uniform(0.5,1,k)/np.arange(1,k+1)
    x = np.linspace(0,2*math.pi,n,endpoint=False)
    s = sum(a*math.sin(f+1)*0 for f,a in zip(range(k),am))
    s = sum(a*np.sin((i+1)*x+p) for i,(a,p) in enumerate(zip(am,ph)))
    return ((s-s.min())/(s.ptp()+1e-9)).astype(np.float32)

# ---------------------------------------------------------------- 程序化纹理
def craterize(tex, seed, n=500, strength=0.5, maxr=6):
    h,w,_ = tex.shape
    rng = np.random.default_rng(seed)
    out = tex.copy()
    for _ in range(n):
        cy = rng.uniform(0.03*h,0.97*h); cx = rng.uniform(0,w)
        r = rng.uniform(1.2,maxr)*(0.5+0.5*rng.random())
        stretch = 1.0/max(math.cos((cy/h-0.5)*math.pi), 0.3)
        R = int(r*3)+2
        y0,y1 = int(cy-R), int(cy+R+2)
        x0,x1 = int(cx-R*stretch), int(cx+R*stretch+2)
        ys = np.arange(max(y0,0),min(y1,h)); xs = np.arange(max(x0,0),min(x1,w))
        if len(ys)==0 or len(xs)==0: continue
        yy,xx = np.meshgrid(ys,xs,indexing='ij')
        dxw = np.minimum(np.abs(xx-cx), w-np.abs(xx-cx))/stretch
        d = np.hypot(dxw, yy-cy)
        rim = np.exp(-((d-r)/(0.30*r))**2); bowl = np.clip(1-d/max(r,1e-3),0,1)**1.6
        sh = strength*(0.85*rim-0.8*bowl)*(0.6+0.8*rng.random())
        out[max(y0,0):min(y1,h), max(x0,0):min(x1,w)] = np.clip(
            out[max(y0,0):min(y1,h), max(x0,0):min(x1,w)] + sh[...,None], 0, 1)
    return out

def make_moon_tex(base_rgb, seed, craters=400, strength=0.5, maxr=6, blotch=0.25):
    h,w = 512,1024
    n = fbm_2d(w,h,seed,(4,8,16,32))
    tex = np.asarray(base_rgb,np.float32)[None,None,:]*np.ones((h,w,1),np.float32)
    tex *= (1-blotch+2*blotch*n)[...,None]
    return craterize(tex, seed+1, craters, strength, maxr)

def make_pluto():
    h,w = 1024,2048
    n1 = fbm_2d(w,h,71,(4,9,20,44)); n2 = fbm_2d(w,h,72,(6,14,30))
    lat = np.linspace(1,-1,h)[:,None]*np.ones((1,w))
    tex = np.array([0.72,0.55,0.42])[None,None,:]*np.ones((h,w,1),np.float32)
    tex *= (0.82+0.35*n1[...,None])
    lon = np.tile(np.arange(w)[None,:],(h,1))/w*360-180
    band = np.exp(-((lat*2.6+0.15)**2))*np.clip(np.cos((lon+40)/180*math.pi)+0.35,0,1)
    tex = tex*(1-0.8*band[...,None])+np.array([0.24,0.13,0.10])[None,None,:]*band[...,None]
    hx,hy = 180/360*w, 0.42*h
    yy,xx = np.mgrid[0:h,0:w]
    dx = np.minimum(np.abs(xx-hx), w-np.abs(xx-hx))/(w*0.10); dy=(yy-hy)/(h*0.20)
    heart = np.exp(-((dx**2+dy**2)**1.1))*(0.7+0.3*n2)
    tex = tex*(1-0.85*heart[...,None])+np.array([0.98,0.93,0.86])[None,None,:]*heart[...,None]
    tex *= (0.9+0.18*n2[...,None])
    return craterize(np.clip(tex,0,1), 77, 260, 0.4, 5)

def make_charon_color():
    """把新视野号卡戎灰度地图上色：灰色地形 + 北极暗红（魔多斑），未成像南极补灰。"""
    src = Image.open(os.path.join(TEX,'charon_map.png')).convert('L')
    a = np.asarray(src,np.float32)/255.0
    h,w = a.shape
    # 南极未成像（黑）→ 用相邻均值 + 噪声补
    col_mean = a[h//3:int(h*0.62)].mean()
    fill = col_mean*0.92 + (fbm_2d(w,h,55,(6,14,30))-0.5)*0.25
    a = np.where(a < 0.04, fill, a)
    a = np.clip(a,0,1)
    tex = np.zeros((h,w,3),np.float32)
    g2r = np.array([1.00,0.97,0.92])   # 亮
    g1r = np.array([0.52,0.50,0.47])   # 中
    g0r = np.array([0.28,0.24,0.21])   # 暗
    t = a[...,None]
    tex = np.where(t>0.5, g1r[None,None,:]+(g2r-g1r)[None,None,:]*((a-0.5)/0.5)[...,None],
                   g0r[None,None,:]+(g1r-g0r)[None,None,:]*(a/0.5)[...,None])
    # 北极偏红（真实：托林沉积）
    lat = np.linspace(1,-1,h)[:,None]
    mordor = np.exp(-((lat-0.82)*5.0)**2)
    tex = tex*(1-0.55*mordor[...,None]) + np.array([0.45,0.20,0.13])[None,None,:]*mordor[...,None]
    return np.clip(tex,0,1)

def make_sedna():
    h,w = 512,1024
    n1 = fbm_2d(w,h,91,(4,9,20,44)); n2 = fbm_2d(w,h,92,(10,22))
    tex = np.array([0.48,0.11,0.06])[None,None,:]*np.ones((h,w,1),np.float32)
    tex *= (0.72+0.5*n1[...,None])
    tex += (np.array([0.72,0.28,0.16])[None,None,:]-tex)*np.clip(n2-0.55,0,1)[...,None]*0.5
    return craterize(np.clip(tex,0,1), 93, 150, 0.30, 5)

def make_ceres():
    return make_moon_tex([0.52,0.50,0.47], 101, 700, 0.55, 6, 0.2)

def make_sunproc():
    h,w = 1024,2048
    n = fbm_2d(w,h,111,(24,48,96,192)); n2 = fbm_2d(w,h,112,(96,192))
    tex = np.ones((h,w,3),np.float32)
    tex[...,1] = 0.86+0.10*n; tex[...,2] = 0.55+0.25*n*n2
    return np.clip(tex,0,1)

def make_ring_profile():
    n = 1024; x = np.linspace(0,1,n).astype(np.float32)
    fine = fbm_1d(n,120,90)
    a = np.zeros(n,np.float32)
    a += np.clip(x*6-0.35,0,1)*0.25*(x<0.32)
    a += 0.95*np.exp(-((x-0.48)/0.16)**2)
    a += 0.55*np.exp(-((x-0.83)/0.10)**2)
    a *= (1-0.93*np.exp(-((x-0.675)/0.018)**2))
    a *= (1-0.85*np.exp(-((x-0.90)/0.006)**2))
    a *= (0.75+0.25*fine); a = np.clip(a,0,1)
    col = np.clip(np.stack([0.86+0.10*fine,0.78+0.10*fine,0.66+0.12*fine],-1),0,1)
    return a, col.astype(np.float32)

# ---------------------------------------------------------------- 纹理加载
def load_tex(name, fallback=None, maxw=4096):
    p = os.path.join(TEX,name)
    try:
        im = Image.open(p).convert('RGB'); im.load()
        if im.width > maxw:
            im = im.resize((maxw,int(im.height*maxw/im.width)), Image.LANCZOS)
        a = np.asarray(im,np.float32)/255.0
        return (a**2.2).astype(np.float32)
    except Exception:
        if fallback is None: return np.full((512,1024,3),0.5,np.float32)
        return np.clip(fallback(),0,1).astype(np.float32)

# ---------------------------------------------------------------- 场景几何
RSCALE = 60.0
RSUN   = 6.0     # 真实应为 109 地球半径，压缩（片头已声明）

def orbit_pos(AU, lam, z=0.0):
    r = RSCALE*AU**0.45
    return np.array([r*math.cos(lam), r*math.sin(lam), z])

LAM = dict(mercury=0.35, venus=0.72, earth=1.10, mars=1.52, jupiter=2.55,
           saturn=3.00, uranus=3.42, neptune=3.85, pluto=4.25, sedna=4.65)
POS = dict(
    sun    = np.array([0.,0.,0.]),
    mercury= orbit_pos(0.387,LAM['mercury'], 0.40),
    venus  = orbit_pos(0.723,LAM['venus'],  -0.45),
    earth  = orbit_pos(1.000,LAM['earth'],   0.30),
    mars   = orbit_pos(1.524,LAM['mars'],   -0.55),
    jupiter= orbit_pos(5.203,LAM['jupiter'], 0.50),
    saturn = orbit_pos(9.537,LAM['saturn'], -0.65),
    uranus = orbit_pos(19.19,LAM['uranus'],  0.40),
    neptune= orbit_pos(30.07,LAM['neptune'],-0.45),
    pluto  = orbit_pos(39.5, LAM['pluto'],   4.0),
    sedna  = orbit_pos(84.0, LAM['sedna'],  -3.5),
)
MOON_OFF = POS['earth'] + np.array([-9.5,5.5,2.2])
CHARON_R = 1.55
PLUTO_DIR = normalize(POS['pluto']-POS['neptune'])
PROX_DIR  = normalize(POS['sedna'])     # 比邻星方向（终段航向）

class Body:
    __slots__ = ('name','pos','radius','tex','axis','e1','e2','lon0','rot',
                 'ring','atmo','night','clouds','limbdark','kind','posfn','gain')
    def __init__(s,name,pos,radius,tex,axis=None,lon0=0.0,rot=0.0,ring=None,
                 atmo=None,night=None,clouds=None,limbdark=0.0,kind='planet',posfn=None,gain=1.0):
        s.name,s.pos,s.radius,s.tex,s.gain = name,np.asarray(pos,float),radius,tex,gain
        s.axis = normalize(axis if axis is not None else [0,0,1.0])
        e1 = np.cross(s.axis,[0,0,1.0])
        if np.linalg.norm(e1)<1e-3: e1 = np.cross(s.axis,[0,1,0.0])
        s.e1 = normalize(e1); s.e2 = np.cross(s.axis,s.e1)
        s.lon0,s.rot,s.ring,s.atmo,s.night,s.clouds = lon0,rot,ring,atmo,night,clouds
        s.limbdark,s.kind,s.posfn = limbdark,kind,posfn
    def center(s,t): return s.posfn(t) if s.posfn is not None else s.pos

def build_bodies():
    L = lambda n,f=None: load_tex(n,f)
    tx = dict(
        sun=L('2k_sun.jpg',make_sunproc), mercury=L('2k_mercury.jpg'),
        venus=L('2k_venus_atmosphere.jpg'),
        earth=L('earth_4k_noclouds.jpg', lambda:L('8k_earth_daymap.jpg', lambda:L('2k_earth_daymap.jpg'))),
        earth_night=L('2k_earth_nightmap.jpg'),
        earth_clouds=L('8k_earth_clouds.jpg', lambda:L('2k_earth_clouds.jpg')),
        moon=L('2k_moon.jpg'), mars=L('2k_mars.jpg'), jupiter=L('2k_jupiter.jpg'),
        saturn=L('2k_saturn.jpg'), uranus=L('2k_uranus.jpg'), neptune=L('2k_neptune.jpg'),
        pluto=None, charon=L('2k_charon.jpg',make_charon_color),
        sedna=L('2k_sedna.jpg',make_sedna), ceres=L('2k_ceres.jpg',make_ceres),
        io=make_moon_tex([0.85,0.78,0.45],31,300,0.5,5),
        europa=make_moon_tex([0.82,0.76,0.66],32,120,0.3,4,0.1),
        ganymede=make_moon_tex([0.55,0.50,0.45],33,350,0.5,5),
        callisto=make_moon_tex([0.42,0.38,0.34],34,500,0.55,5),
        titan=make_moon_tex([0.82,0.56,0.24],35,60,0.2,4,0.08),
    )
    # 冥王星：真实地图 + 对比度拉伸（原图偏暗）
    pl = L('2k_pluto.jpg',make_pluto)
    p1,p99 = np.percentile(pl,[1,99.5])
    tx['pluto'] = np.clip((pl-p1)/(p99-p1),0,1).astype(np.float32)
    del pl
    # 云图：无缝 4k fair_clouds（alpha=覆盖率），行剖面轻度均衡
    try:
        _im = Image.open(os.path.join(TEX,'clouds_4k.png')).convert('RGBA'); _im.load()
        _a = np.asarray(_im,np.float32)/255.0
        cov = _a[...,3]
        prof = np.convolve(cov.mean(axis=1),np.ones(65)/65,mode='same')
        tgt = prof[len(prof)//4:3*len(prof)//4].mean()
        ratio = np.clip(tgt/(prof+1e-4),0.85,1.18)
        cov = np.clip(cov*ratio[:,None],0,1)
        tx['earth_clouds'] = cov[...,None].repeat(3,axis=2).astype(np.float32)
        del _a
    except Exception:
        pass
    # 夜灯图与昼图经度对齐（暴力互相关）
    def _small(x):
        return np.asarray(Image.fromarray((np.clip(x.mean(-1),0,1)*255).astype(np.uint8)).resize((512,256)),np.float32)
    try:
        d_s = _small(tx['earth']); n_s = _small(tx['earth_night'])
        best_s, best_v = 0, -1e18
        for s in range(-256,257):
            v = float((d_s*np.roll(n_s,s,axis=1)).sum())
            if v > best_v: best_v, best_s = v, s
        px = best_s*tx['earth_night'].shape[1]//512
        if abs(px) > 4:
            tx['earth_night'] = np.roll(tx['earth_night'],px,axis=1)
        print('夜灯对齐: 经度偏移', px, 'px')
    except Exception as e:
        print('夜灯对齐跳过:', e)
    ringA,ringC = make_ring_profile()
    sat_ring = dict(inner=9.14*1.18,outer=9.14*2.32,alpha=ringA,color=ringC,thin=False)
    try:
        im = Image.open(os.path.join(TEX,'2k_saturn_ring_alpha.png')).convert('RGBA'); im.load()
        a = np.asarray(im,np.float32)/255.0
        strip = a.mean(axis=0)
        al = np.clip(strip[:,3]/max(strip[:,3].max(),1e-6),0,1)
        if len(al)>64: sat_ring.update(alpha=al.astype(np.float32),
                                       color=(strip[:,:3]**2.2).astype(np.float32))
    except Exception: pass
    ura_ring = dict(inner=3.98*1.62,outer=3.98*2.05,thin=True)

    B = {}
    def add(nm,**kw):
        pos = kw.pop('pos'); radius = kw.pop('radius'); texx = kw.pop('tex')
        B[nm] = Body(nm,pos,radius,texx,**kw)
    add('sun',pos=POS['sun'],radius=RSUN,tex=tx['sun'],lon0=2.4,kind='sun')
    add('mercury',pos=POS['mercury'],radius=0.383,tex=tx['mercury'],lon0=3.6,gain=1.3)
    add('venus',pos=POS['venus'],radius=0.949,tex=tx['venus'],
        axis=rot_z(0.9)@rot_x(math.radians(177))@[0,0,1],lon0=1.0,rot=0.010,
        atmo=dict(col=[1.0,0.93,0.72],k=0.8))
    add('earth',pos=POS['earth'],radius=1.0,tex=tx['earth'],gain=1.15,
        axis=rot_z(0.4)@rot_x(math.radians(23.4))@[0,0,1],lon0=4.6,rot=0.02,
        atmo=dict(col=[0.30,0.55,1.0],k=1.0),night=tx['earth_night'],clouds=tx['earth_clouds'])
    add('moon',pos=MOON_OFF,radius=0.273,tex=tx['moon'],lon0=0.9)
    add('mars',pos=POS['mars'],radius=0.532,tex=tx['mars'],gain=1.25,
        axis=rot_z(2.2)@rot_x(math.radians(25.2))@[0,0,1],lon0=0.5,
        atmo=dict(col=[0.9,0.6,0.45],k=0.35))
    add('jupiter',pos=POS['jupiter'],radius=10.97,tex=tx['jupiter'],
        axis=rot_z(1.2)@rot_x(math.radians(3.1))@[0,0,1],lon0=0.6,rot=0.10,limbdark=0.35)
    add('saturn',pos=POS['saturn'],radius=9.14,tex=tx['saturn'],
        axis=rot_z(0.2)@rot_x(math.radians(26.7))@[0,0,1],lon0=2.3,rot=0.07,
        ring=sat_ring,limbdark=0.30,atmo=dict(col=[0.95,0.88,0.70],k=0.35))
    add('uranus',pos=POS['uranus'],radius=3.98,tex=tx['uranus'],
        axis=rot_z(2.0)@rot_x(math.radians(97.8))@[0,0,1],lon0=1.0,rot=0.05,
        ring=ura_ring,limbdark=0.30,atmo=dict(col=[0.55,0.85,0.9],k=0.5))
    add('neptune',pos=POS['neptune'],radius=3.86,tex=tx['neptune'],gain=1.5,
        axis=rot_z(1.5)@rot_x(math.radians(28.3))@[0,0,1],lon0=2.9,rot=0.06,
        limbdark=0.30,atmo=dict(col=[0.35,0.5,0.95],k=0.5))
    add('pluto',pos=POS['pluto'],radius=0.187,tex=tx['pluto'],gain=1.6,lon0=2.60,
        axis=rot_z(1.0)@rot_x(math.radians(120))@[0,0,1],rot=0.015)
    def charon_pos(t):
        a = 0.12*t+2.6
        return POS['pluto']+np.array([math.cos(a),math.sin(a),0.32])*CHARON_R
    add('charon',pos=POS['pluto'],radius=0.095,tex=tx['charon'],lon0=2.0,posfn=charon_pos,gain=2.0)
    add('sedna',pos=POS['sedna'],radius=0.078,tex=tx['sedna'],lon0=1.5,rot=0.02,gain=1.5)
    for i,(nm,r_,dd) in enumerate([('io',0.286,16.5),('europa',0.245,21.0),
                                    ('ganymede',0.413,26.5),('callisto',0.378,34.0)]):
        ang = 0.9+i*1.7
        add(nm,pos=POS['jupiter']+np.array([math.cos(ang),math.sin(ang),0.10])*dd,
            radius=r_,tex=tx[nm],lon0=i*1.9)
    add('titan',pos=POS['saturn']+np.array([math.cos(1.9),math.sin(1.9),0.18])*22.0,
        radius=0.404,tex=tx['titan'],lon0=0.3,atmo=dict(col=[0.95,0.65,0.30],k=0.9))
    return B

# ---------------------------------------------------------------- 星空
def kelvin_rgb(T):
    T = T/100.
    if T<=66: r=255.; g=99.47*math.log(max(T,1.))-161.1
    else: r=329.7*(T-60)**-0.1332; g=288.1*(T-60)**-0.0755
    if T>=66: b=255.
    elif T<=19: b=0.
    else: b=138.5*math.log(max(T-10,1.))-305.
    c = np.clip(np.array([r,g,b]),0,255)/255.
    return c/max(c.max(),1e-6)

def build_sky():
    if os.path.exists(SKYCACHE):
        z = np.load(SKYCACHE)
        return z['mw'], (z['stars'],z['flux'],z['scol']), z['prox'], z['acenA'], z['acenB']
    SW,SH = 2048,1024
    rng = np.random.default_rng(SEED)
    lon = (np.arange(SW)+0.5)/SW*2*math.pi-math.pi
    lat = math.pi/2-(np.arange(SH)+0.5)/SH*math.pi
    LON,LAT = np.meshgrid(lon,lat)
    vx = np.cos(LAT)*np.cos(LON); vy = np.cos(LAT)*np.sin(LON); vz = np.sin(LAT)
    g = normalize([-0.05,-0.12,0.99])
    b = vx*g[0]+vy*g[1]+vz*g[2]
    band = np.exp(-(np.arcsin(np.clip(b,-1,1))/0.115)**2)
    n1 = fbm_2d(SW,SH,201,(3,7,15,31)); n2 = fbm_2d(SW,SH,202,(5,11,23))
    glow = band*(0.55+0.85*n1)
    glow *= (1-1.25*np.clip(np.clip(n2-0.52,0,1)*2.4,0,0.92))
    warm = np.array([1.0,0.93,0.82]); cool = np.array([0.75,0.82,1.0])
    mixc = (0.35+0.65*n2)[...,None]
    col = warm[None,None,:]*mixc+cool[None,None,:]*(1-mixc)
    mw = (glow[...,None]*col*0.085 + (band*0.042)[...,None]*cool[None,None,:])
    mw = np.clip(mw,0,0.5).astype(np.float32)
    N = 60000
    dirs = rng.normal(size=(N,3)); dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    mag = 6.8-7.3*rng.random(N)**3.2
    flux = (10**(-0.33*(mag+1.3))).astype(np.float32)
    temps = rng.choice([3200,4200,5000,5800,6600,9000,15000],N,
                       p=[.18,.22,.20,.18,.12,.07,.03])
    scol = np.stack([kelvin_rgb(T) for T in temps]).astype(np.float32)
    prox = PROX_DIR.astype(np.float32)
    ax = normalize(np.cross(prox,[0,0,1.0]))
    ay = normalize(np.cross(ax,prox))
    acenA = normalize(prox+ax*math.tan(math.radians(2.0))).astype(np.float32)
    acenB = normalize(prox+ax*math.tan(math.radians(2.0))+ay*math.tan(math.radians(0.22))).astype(np.float32)
    np.savez_compressed(SKYCACHE,mw=mw,stars=dirs.astype(np.float32),flux=flux,
                        scol=scol,prox=prox,acenA=acenA,acenB=acenB)
    return mw,(dirs.astype(np.float32),flux,scol),prox,acenA,acenB

def build_belt():
    rng = np.random.default_rng(SEED+9)
    n = 2600
    rr = RSCALE*rng.uniform(2.15,3.25,n)**0.45
    th = rng.uniform(0,2*math.pi,n)
    z  = rng.normal(0,2.2,n)
    size = np.clip((rng.pareto(2.6,n)+0.06)*0.16,0.02,1.4)
    belt = np.stack([rr*np.cos(th),rr*np.sin(th),z,size],1).astype(np.float32)
    n2 = 900
    rr2 = RSCALE*rng.uniform(38,46,n2)**0.45
    th2 = rng.uniform(0,2*math.pi,n2); z2 = rng.normal(0,5.0,n2)
    sz2 = np.clip((rng.pareto(2.6,n2)+0.06)*0.14,0.02,1.0)
    kuiper = np.stack([rr2*np.cos(th2),rr2*np.sin(th2),z2,sz2],1).astype(np.float32)
    return belt,kuiper

# ---------------------------------------------------------------- 相机路径
def tangent(p):
    r = normalize(np.array([p[0],p[1],0.0])); return np.cross([0,0,1.0],r)

def build_path(B):
    P = POS
    mdir = normalize(P['mercury']-P['sun']); vdir = normalize(P['venus']-P['mercury'])
    edir = normalize(P['earth']-P['venus']); mardir = normalize(P['mars']-P['earth'])
    jdir = normalize(P['jupiter']-P['mars']); sdir = normalize(P['saturn']-P['jupiter'])
    udir = normalize(P['uranus']-P['saturn']); ndir = normalize(P['neptune']-P['uranus'])
    pldir = PLUTO_DIR; sedir = normalize(P['sedna']-P['pluto'])
    charon = B['charon'].posfn
    mid_pc = lambda t: (POS['pluto']+charon(t))/2
    K = []
    def add(t,cam,tgt,fov): K.append((t,np.asarray(cam,float),np.asarray(tgt,float),fov))
    def near(p,toward,d,z=0.,lat=0.):
        v = normalize(np.asarray(toward,float)); t = tangent(p)
        return p+v*d+np.array([0,0,z])+t*lat

    # 太阳：从木星-土星之间的空旷方位角(158°)接近——回望视野内无任何行星
    sapp = np.array([math.cos(2.76), math.sin(2.76), 0.0])
    add(0.0, P['sun']+sapp*13.0+[0,0,4.2], P['sun'], 72)
    add(2.8, P['sun']+sapp*15.0+np.array([1.5,1.2,5.4]), P['sun'], 66)
    # 离场修复：太阳不再被画面边缘切割、不再瞬间甩出——
    # 1) 直线加速后退期间注视点保持在太阳（盘面始终居中、缩至≤185px）
    # 2) 之后经大弧绕向水星侧后方完成摇镜，太阳以小盘从画缘平滑滑出
    mtan = tangent(P['mercury'])
    dApp = normalize(-mdir*0.5 + mtan*0.87)          # 水星侧后 60° 相位方向（下弦偏盈）
    ub = normalize(P['mercury']+dApp*20.0 - P['sun'])
    smid = normalize(sapp+ub)                          # 弧线中点方向（绕日 ~55°）
    add(4.3, P['sun']+sapp*40.0+np.array([2,1.8,10.0]), P['sun'], 58)      # 居中缩小
    add(5.4, P['sun']+smid*40.0+np.array([1,1,6.0]), P['mercury'], 54)     # 弧顶摇向水星
    add(6.3, P['mercury']+dApp*20.0+np.array([0,0,1.0]), P['mercury'],52)  # 侧后 60° 相位
    add(7.0, P['mercury']+dApp*11.0+np.array([0,0,0.8]), P['mercury'],50)
    add(7.5, P['mercury']+dApp*5.5+np.array([0,0,0.7]), P['mercury'],48)
    add(8.0, P['mercury']+mtan*2.6+np.array([0,0,0.9]), P['mercury'],48)   # 侧向半相位特写
    add(10.2, near(P['mercury'],tangent(P['mercury']),1.28,0.28,0.4), P['mercury'],44)
    add(12.4, near(P['mercury'],tangent(P['mercury']),1.9,0.42,0.7), P['mercury'],44)  # 停留
    add(14.6, near(P['mercury'],vdir,3.2,0.7,-0.8), P['venus'],48)
    # 金星
    add(16.8, near(P['venus'],-vdir,6.4,0.7,2.0), P['venus'],48)
    add(18.8, near(P['venus'],tangent(P['venus']),3.0,0.9,0.9), P['venus'],43)
    add(20.2, near(P['venus'],tangent(P['venus']),3.6,1.0,0.7), P['venus'],43)        # 停留
    add(22.0, near(P['venus'],edir,6.5,0.9,-1.6), P['earth'],48)
    # 地球 & 月球
    add(24.0, near(P['earth'],-edir,8.0,1.6,2.4), P['earth'],48)
    add(25.9, near(P['earth'],tangent(P['earth']),3.4,1.1,-0.3), P['earth'],42)
    add(27.4, P['earth']+tangent(P['earth'])*30.0+np.array([0,0,8.5])-edir*7.0,
        (P['earth']+MOON_OFF)/2, 50)                     # 地月同框
    add(28.6, P['earth']+tangent(P['earth'])*26.0+np.array([0,0,7.5])-edir*5.5,
        (P['earth']+MOON_OFF)/2, 48)                     # 推近保持同框
    add(29.9, P['earth']+tangent(P['earth'])*40.0+np.array([0,0,9.0]),
        (P['earth']+MOON_OFF)/2, 46)                     # 径直后退，地月渐小、月球不乱入
    add(31.2, P['earth']+tangent(P['earth'])*42.0+np.array([0,0,8.6])+edir*2.0,
        P['mars'], 46)                                   # 在远处缓缓摇向火星
    # 32.6 起为火星接近段（保持不变）
    # 火星
    add(32.6, near(P['mars'],-mardir,3.8,0.55,1.4), P['mars'],48)
    add(34.2, near(P['mars'],tangent(P['mars']),1.7,0.42,0.5), P['mars'],43)
    add(35.6, near(P['mars'],tangent(P['mars']),2.4,0.55,0.9), P['mars'],43)  # 停留
    add(37.4, near(P['mars'],jdir,4.5,0.9,-1.5), P['jupiter'],50)
    # 火→木巡航（三段渐变，木星渐大入画）
    add(39.6, near(P['mars'],jdir,16.0,3.0,-5.0), P['jupiter'],52)
    add(42.2, near(P['jupiter'],-jdir,68.0,8.0,18.0), P['jupiter'],48)
    # 木星
    add(44.6, near(P['jupiter'],-jdir,40.0,5.0,10.0), P['jupiter'],46)
    add(46.4, near(P['jupiter'],tangent(P['jupiter']),34.0,4.0,3.0), P['jupiter'],44)
    add(48.6, near(P['jupiter'],tangent(P['jupiter']),30.0,5.2,6.0), P['jupiter'],44)  # 停留
    add(50.6, near(P['jupiter'],sdir,50.0,8.0,-12.0), P['saturn'],48)
    # 土星
    add(52.6, near(P['saturn'],-sdir,52.0,10.0,18.0), P['saturn'],46)
    add(55.0, near(P['saturn'],tangent(P['saturn']),34.0,12.5,6.0), P['saturn'],44)
    add(57.0, near(P['saturn'],tangent(P['saturn']),31.0,13.2,7.5), P['saturn'],44)   # 停留
    add(59.0, near(P['saturn'],udir,55.0,10.0,-16.0), P['uranus'],48)
    # 天王星
    add(61.0, near(P['uranus'],-udir,18.0,3.0,6.0), P['uranus'],46)
    add(62.8, near(P['uranus'],tangent(P['uranus']),12.5,2.6,1.8), P['uranus'],43)
    add(64.2, near(P['uranus'],tangent(P['uranus']),13.5,2.9,2.4), P['uranus'],43)    # 停留
    add(66.0, near(P['uranus'],ndir,20.0,4.0,-6.0), P['neptune'],48)
    # 海王星
    add(67.8, near(P['neptune'],-ndir,18.0,3.4,6.0), P['neptune'],46)
    add(69.6, near(P['neptune'],tangent(P['neptune']),12.6,2.8,1.9), P['neptune'],43)
    add(71.0, near(P['neptune'],tangent(P['neptune']),13.4,3.1,2.4), P['neptune'],43) # 停留
    add(72.8, near(P['neptune'],pldir,22.0,4.5,-6.5), P['pluto'],48)
    # 冥王星 & 卡戎
    prad = normalize(P['pluto'])
    ptan = tangent(P['pluto'])
    baim = P['pluto']+normalize(charon(76.5)-P['pluto'])*0.5      # 双星段统一瞄准点
    add(73.9, P['pluto']-prad*45.0+ptan*5.0+np.array([0,0,2.0]), P['pluto'],48)   # 减速阶梯
    add(74.8, P['pluto']-prad*16.0+ptan*3.2+np.array([0,0,1.6]), baim,44)
    add(75.8, P['pluto']-prad*7.5+ptan*2.7+np.array([0,0,1.4]), baim,42)
    add(76.8, P['pluto']-prad*6.4+ptan*3.2+np.array([0,0,1.2]), baim,42)          # 双星同框
    add(77.8, P['pluto']-prad*6.8+ptan*4.2+np.array([0,0,1.0]), baim,42)
    add(79.6, P['pluto']-prad*7.6+ptan*5.6+np.array([0,0,0.6]), P['sedna'],46)
    # 赛德娜：单调减速接近并停留（始终向阳侧）
    srad = normalize(P['sedna'])
    stang = tangent(P['sedna'])
    add(80.2, P['sedna']-srad*70.0+stang*3.0+np.array([0,0,1.5]), P['sedna'],48)    # 减速阶梯
    add(80.9, P['sedna']-srad*28.0+stang*1.5+np.array([0,0,0.9]), P['sedna'],46)
    add(81.6, P['sedna']-srad*9.0+stang*0.8+np.array([0,0,0.50]), P['sedna'],44)
    add(82.4, P['sedna']-srad*3.2+stang*0.5+np.array([0,0,0.35]), P['sedna'],42)
    add(83.4, P['sedna']-srad*1.10+stang*0.30+np.array([0,0,0.22]), P['sedna'],40)
    add(84.0, P['sedna']-srad*0.72+stang*0.20+np.array([0,0,0.16]), P['sedna'],40)  # 最近
    add(84.8, P['sedna']-srad*0.95-stang*0.10+np.array([0,0,0.20]), P['sedna'],40)
    add(85.4, P['sedna']-srad*1.9-stang*0.15+np.array([0,0,0.32]), P['sedna']+PROX_DIR*20.0,42)  # 先退再转
    add(85.9, near(P['sedna'],PROX_DIR,2.2,0.3,-0.4), PROX_DIR*600,46)
    # 终章
    add(87.0, P['sedna']+PROX_DIR*7+[0,0,1.2], PROX_DIR*600,34)
    add(90.5, P['sedna']+PROX_DIR*11+[0,0,1.6], PROX_DIR*600,24)
    add(93.5, P['sedna']+PROX_DIR*13.5+[0,0,1.9], PROX_DIR*600,16)
    add(96.0, P['sedna']+PROX_DIR*14.5+[0,0,2.0], PROX_DIR*600,14)

    # 长航段插入中间关键帧：速度更均匀，切换天体不再突兀
    K2 = [K[0]]
    for i in range(1,len(K)):
        t0,c0,g0,f0 = K[i-1]; t1,c1,g1,f1 = K[i]
        dist = float(np.linalg.norm(c1-c0)); dt = t1-t0
        n = 3 if dist>26 else (2 if dist>12 else (1 if dist>9 and dt>2.4 else 0))
        for j in range(1,n+1):
            a = j/(n+1)
            K2.append((t0+dt*a, c0*(1-a)+c1*a, g0*(1-a)+g1*a, f0*(1-a)+f1*a))
        K2.append(K[i])
    ts0   = np.array([k[0] for k in K])          # 原始关键帧：注视轨道（不加密，弦上不会出现中间点）
    tgts0 = np.array([k[2] for k in K])
    K = K2
    ts   = np.array([k[0] for k in K])
    cams = np.array([k[1] for k in K]); tgts = np.array([k[2] for k in K])
    fovs = np.array([k[3] for k in K])
    return ts,cams,tgts,fovs,ts0,tgts0

AU_KEY_T = [0,2.8,5.6,8,10.2,12.4,14.6,16.8,18.8,20.2,22,24,26.2,28.2,29.4,31,
            32.6,34.2,35.6,37.4,39.6,42.2,44.6,46.4,48.6,50.6,52.6,55,57,59,
            61,62.8,64.2,66,67.8,69.6,71,72.8,74.8,76.2,77.8,79.6,80.8,85.4,87,90.5,93.5,96]
AU_KEY_V = [0.02,0.02,0.05,0.39,0.39,0.39,0.39,0.72,0.72,0.72,0.72,1.0,1.0,1.0,1.0,1.0,
            1.52,1.52,1.52,1.52,2.7,3.5,5.2,5.2,5.2,5.2,9.5,9.5,9.5,9.5,
            19.2,19.2,19.2,19.2,30.1,30.1,30.1,30.1,39.5,39.5,39.5,39.5,84,84,84,84,84,84]

# ---------------------------------------------------------------- 渲染
def sample_equirect(tex,u,v):
    h,w,_ = tex.shape
    u = np.mod(u,w).astype(np.float32)
    u = np.where(u>=w, u-w, u)          # float32 精度保护
    v = np.clip(v,0,h-1.001)
    x0 = np.floor(u).astype(np.int32); y0 = np.floor(v).astype(np.int32)
    fx = (u-x0)[...,None]; fy = (v-y0)[...,None]
    x1 = (x0+1)%w; y1 = np.minimum(y0+1,h-1)
    c00 = tex[y0,x0]; c10 = tex[y0,x1]; c01 = tex[y1,x0]; c11 = tex[y1,x1]
    return (c00*(1-fx)+c10*fx)*(1-fy)+(c01*(1-fx)+c11*fx)*fy

VIG = None
def vig_mask():
    global VIG
    if VIG is None:
        yy,xx = np.mgrid[0:H,0:W].astype(np.float32)
        dd = np.sqrt(((xx-W/2)/(W/2))**2+((yy-H/2)/(H/2))**2)
        VIG = (1-0.22*np.clip(dd-0.55,0,1)**1.8)[...,None].astype(np.float32)
    return VIG

def aces(x):
    a,b,c,d,e = 2.51,0.03,2.43,0.59,0.14
    return np.clip(x*(a*x+b)/(x*(c*x+d)+e),0,1)**(1/2.2)

def post_process(img):
    x = img.astype(np.float32)
    hi = np.clip(img.max(-1)-1.02,0,None)
    if hi.max() > 0.01:
        im = Image.fromarray(np.clip(hi*120,0,255).astype(np.uint8))
        im = im.resize((W//4,H//4),Image.BILINEAR).filter(ImageFilter.GaussianBlur(5))
        im = im.resize((W,H),Image.BILINEAR).filter(ImageFilter.GaussianBlur(2))
        x = x + (np.asarray(im,np.float32)/255.0)[...,None]*0.20
    x = aces(x*1.02)
    x *= vig_mask()
    x += np.random.normal(0,0.0028,x.shape).astype(np.float32)
    return np.clip(x*255+0.5,0,255).astype(np.uint8)

class Renderer:
    def __init__(s):
        s.B = build_bodies()
        s.mw,s.stars,s.prox,s.acenA,s.acenB = build_sky()
        s.path = build_path(s.B)
        s.ts0,s.tgts0 = s.path[4],s.path[5]
        s.belt,s.kuiper = build_belt()

    def camera(s,t):
        ts,cams,tgts,fovs = s.path[0],s.path[1],s.path[2],s.path[3]
        E = hermite_limited(ts,cams,t)
        ts0,tgts0 = s.ts0,s.tgts0
        i = int(np.clip(np.searchsorted(ts0,t)-1,0,len(ts0)-2))
        u = (t-ts0[i])/max(ts0[i+1]-ts0[i],1e-9)
        A,B = tgts0[i],tgts0[i+1]
        fov = cr_scalar(ts,fovs,t)
        if np.linalg.norm(B-A) <= 1e-6:
            f = normalize(A-E)                       # 保持段：精确瞄准行星中心
        else:
            w = u*u*(3-2*u)                          # 摇镜缓入缓出
            fA = normalize(A-E); fB = normalize(B-E)
            cv = max(-1.0,min(1.0,float(fA@fB))); ang = math.acos(cv)
            if ang < 0.5:
                g = (1-w)*fA+w*fB                       # 小角度：nlerp 精确终止于 fB（无段边界跳变）
                f = g/max(float(np.linalg.norm(g)),1e-12)
            elif True:
                # 转轴按"日心方向夹角"整段恒定（避免帧间阈值翻转）
                uA = normalize(A) if np.linalg.norm(A)>1e-6 else -normalize(E)
                uB = normalize(B)
                angH = math.acos(max(-1.0,min(1.0,float(uA@uB))))
                if angH > 2.45:                      # 大范围摇镜：绕黄道轴正向，扫过远侧空旷天区
                    ax = np.array([0.,0.,1.])
                else:
                    c = np.cross(fA,fB); nc = float(np.linalg.norm(c))
                    ax = normalize(c) if nc > 0.1 else np.array([0.,0.,1.])
                th = w*ang
                f = normalize(fA*math.cos(th)+np.cross(ax,fA)*math.sin(th)
                              +ax*float(ax@fA)*(1-math.cos(th)))
        uph = np.array([0,0,1.0])
        if abs(f@uph)>0.985: uph = np.array([0,1,0.0])
        r = normalize(np.cross(f,uph)); u = np.cross(r,f)
        return E,f,r,u,math.tan(math.radians(fov)*0.5)

    def render(s,t):
        E,f,r,u,tanf = s.camera(t)
        aspect = W/H
        yy,xx = np.mgrid[0:H:1,0:W:1].astype(np.float32)
        px = ((xx+0.5-W/2)/(H/2))*tanf
        py = ((H/2-(yy+0.5))/(H/2))*tanf
        d = (f[0]+r[0]*px+u[0]*py, f[1]+r[1]*px+u[1]*py, f[2]+r[2]*px+u[2]*py)
        dn = 1.0/np.sqrt(d[0]**2+d[1]**2+d[2]**2)
        d = (d[0]*dn,d[1]*dn,d[2]*dn)
        img = s.sky_background(d)
        s.draw_stars(img,E,f,r,u,tanf,t)
        s.draw_belt(img,E,f,r,u,tanf)
        order = sorted(s.B.values(), key=lambda b:-np.dot(b.center(t)-E,b.center(t)-E))
        for b in order:
            rel = b.center(t)-E; dist = np.linalg.norm(rel)
            if b.kind=='sun':
                s.draw_sun(img,E,f,r,u,tanf,b,t); continue
            if dist <= b.radius*1.06: continue
            ang = b.radius/dist
            if ang > 0.0022 and (rel@f) > 0:
                s.draw_sphere(img,E,f,r,u,tanf,b,t)
                if b.ring is not None and ang > 0.004:
                    s.draw_ring(img,E,f,r,u,tanf,b,t)
        return post_process(img)

    def sky_background(s,d):
        lon = np.arctan2(d[1],d[0]); lat = np.arcsin(np.clip(d[2],-1,1))
        sw,sh = s.mw.shape[1],s.mw.shape[0]
        uu = (lon/(2*math.pi)+0.5)*sw; vv = (0.5-lat/math.pi)*sh
        return sample_equirect(s.mw,uu,vv).astype(np.float32)

    def draw_stars(s,img,E,f,r,u,tanf,t):
        dirs,flux,cols = s.stars
        zc = dirs@f
        idx = np.where(zc>0.35)[0]
        if len(idx)==0: pass
        else:
            xc = dirs[idx]@r; yc = dirs[idx]@u
            sx = (xc/zc[idx])/tanf*(H/2)+W/2
            sy = -(yc/zc[idx])/tanf*(H/2)+H/2
            gg = np.array([[0.30,0.70,1.0,0.70,0.30],
                           [0.70,1.40,1.9,1.40,0.70],
                           [1.00,1.90,3.20,1.90,1.00],
                           [0.70,1.40,1.9,1.40,0.70],
                           [0.30,0.70,1.0,0.70,0.30]],np.float32)
            for j in range(len(idx)):
                x,y = int(sx[j]),int(sy[j])
                if not(2<x<W-3 and 2<y<H-3): continue
                bri = flux[idx[j]]; col = cols[idx[j]]
                if bri < 0.12:
                    img[y,x] += col*bri*2.0
                else:
                    img[y-2:y+3,x-2:x+3] += gg[...,None]*col[None,None,:]*min(bri*1.35,4.2)*0.55
        # 终章为纯星空背景（比邻星/半人马座α不做特殊绘制）

    def draw_special_star(s,img,dirn,E,f,r,u,tanf,col,k,flare=False):
        zc = dirn@f
        if zc<=0.30: return
        xc = dirn@r; yc = dirn@u
        sx = (xc/zc)/tanf*(H/2)+W/2; sy = -(yc/zc)/tanf*(H/2)+H/2
        x,y = int(sx),int(sy)
        if not(6<x<W-7 and 6<y<H-7): return
        gg = np.array([[0.0,0.16,0.30,0.16,0.0],
                       [0.16,0.55,0.95,0.55,0.16],
                       [0.30,0.95,2.60,0.95,0.30],
                       [0.16,0.55,0.95,0.55,0.16],
                       [0.0,0.16,0.30,0.16,0.0]],np.float32)
        if k < 0.35:                                   # 点源状态：仅一个亮点
            v = col*k*1.5
            img[y,x] += v
            if k > 0.18:
                img[y-1,x] += v*0.35; img[y+1,x] += v*0.35
                img[y,x-1] += v*0.35; img[y,x+1] += v*0.35
            return
        img[y-2:y+3,x-2:x+3] += gg[...,None]*col[None,None,:]*k
        if flare and k>0.4:
            L = int(min(480,70+150*k))
            xs = np.arange(-L,L+1)
            prof = np.exp(-(xs/(L*0.20))**2)*0.8*k
            x0f,x1f = max(x-L,0),min(x+L+1,W)
            if x1f>x0f:
                seg = prof[x0f-(x-L):x1f-(x-L)]
                img[y,x0f:x1f] += col[None,:]*seg[:,None]
                img[y-1,x0f:x1f] += col[None,:]*seg[:,None]*0.35
                img[y+1,x0f:x1f] += col[None,:]*seg[:,None]*0.35
            Lv = L//2
            ys2 = np.arange(-Lv,Lv+1)
            prof2 = np.exp(-(ys2/(Lv*0.14))**2)*0.42*k
            y0f,y1f = max(y-Lv,0),min(y+Lv+1,H)
            if y1f>y0f:
                seg2 = prof2[y0f-(y-Lv):y1f-(y-Lv)]
                img[y0f:y1f,x] += col[None,:]*seg2[:,None]
            i0,i1 = max(x-40,0),min(x+41,W); j0,j1 = max(y-40,0),min(y+41,H)
            ys3,xs3 = np.mgrid[j0:j1,i0:i1]
            dd = np.sqrt((xs3-x)**2+(ys3-y)**2)+1e-3
            g2 = np.exp(-(dd/8.0)**2)*2.0*k+np.exp(-(dd/22.0)**2)*0.45*k
            img[j0:j1,i0:i1] += g2[...,None]*col[None,None,:]

    def draw_belt(s,img,E,f,r,u,tanf):        # 仅保留柯伊伯带远景点缀
        for part,bright0 in ((s.kuiper,0.45),):
            rel = part[:,:3]-E[None,:]
            zc = rel@f
            idx = np.where(zc>1.0)[0]
            if len(idx)==0: continue
            rel2 = rel[idx]; z2 = zc[idx]
            xc = rel2@r; yc = rel2@u
            sx = (xc/z2)/tanf*(H/2)+W/2
            sy = -(yc/z2)/tanf*(H/2)+H/2
            sz = part[idx,3]
            ok = (sx>0)&(sx<W-1)&(sy>0)&(sy<H-1)&(z2<120)
            colbase = np.array([0.76,0.69,0.61],np.float32)
            for j in np.where(ok)[0]:
                x,y = int(sx[j]),int(sy[j])
                fade = 0.35+0.65*min(1.0,9.0/z2[j])
                b = bright0*fade*min(0.95,0.30+sz[j]*2.6)
                col = colbase*b
                img[y,x] = np.maximum(img[y,x],col)
                if b>0.4:
                    img[y,min(x+1,W-1)] = np.maximum(img[y,min(x+1,W-1)],col*0.5)
                    img[min(y+1,H-1),x] = np.maximum(img[min(y+1,H-1),x],col*0.5)

    def draw_sphere(s,img,E,f,r,u,tanf,b,t,chunk=150):
        C = b.center(t); R = b.radius
        rel = C-E; dist = np.linalg.norm(rel)
        if dist<=R*1.05: return
        cz = rel@f
        if cz<0.1: return
        rad_px = math.tan(math.asin(min(1.0,R/dist)))/tanf*(H/2)
        cx = ((rel@r)/cz)/tanf*(H/2)+W/2
        cy = -((rel@u)/cz)/tanf*(H/2)+H/2
        pad = rad_px*1.28+6
        i0,i1 = max(int(cx-pad),0),min(int(cx+pad),W)
        j0,j1 = max(int(cy-pad),0),min(int(cy+pad),H)
        if i1<=i0 or j1<=j0: return
        oc = E-C
        axis,e1,e2 = b.axis,b.e1,b.e2
        lon_t = b.lon0+b.rot*t
        tex = b.tex; htw,hth = tex.shape[1],tex.shape[0]
        ring = b.ring if (b.ring is not None and not b.ring.get('thin')) else None
        for jj0 in range(j0,j1,chunk):
            jj1 = min(jj0+chunk,j1)
            ys,xs = np.mgrid[jj0:jj1,i0:i1]
            pxs = ((xs+0.5-W/2)/(H/2))*tanf
            pys = ((H/2-(ys+0.5))/(H/2))*tanf
            dx = f[0]+r[0]*pxs+u[0]*pys; dy = f[1]+r[1]*pxs+u[1]*pys; dz = f[2]+r[2]*pxs+u[2]*pys
            dl = 1.0/np.sqrt(dx*dx+dy*dy+dz*dz); dx*=dl; dy*=dl; dz*=dl
            bq = oc[0]*dx+oc[1]*dy+oc[2]*dz; cq = oc@oc-R*R
            disc = bq*bq-cq; hit = disc>0
            if not hit.any(): continue
            tq = -bq-np.sqrt(np.where(hit,disc,0)); hit &= tq>0
            if not hit.any(): continue
            Px = E[0]+dx*tq; Py = E[1]+dy*tq; Pz = E[2]+dz*tq
            nx,ny,nz = (Px-C[0])/R,(Py-C[1])/R,(Pz-C[2])/R
            pl = 1.0/np.sqrt(Px*Px+Py*Py+Pz*Pz)
            lx,ly,lz = -Px*pl,-Py*pl,-Pz*pl
            ndl = np.clip(nx*lx+ny*ly+nz*lz,0,1)
            if ring is not None:                       # 环投在行星上的影子
                m = b.axis
                den = lx*m[0]+ly*m[1]+lz*m[2]
                num = (C[0]-Px)*m[0]+(C[1]-Py)*m[1]+(C[2]-Pz)*m[2]
                okd = np.abs(den)>1e-7
                tt = np.where(okd,num/np.where(okd,den,1),-1)
                Qx = Px+lx*tt; Qy = Py+ly*tt; Qz = Pz+lz*tt
                rho = np.sqrt((Qx-C[0])**2+(Qy-C[1])**2+(Qz-C[2])**2)
                inring = (tt>0)&(rho>ring['inner'])&(rho<ring['outer'])
                if inring.any():
                    xidx = np.clip((rho-ring['inner'])/(ring['outer']-ring['inner'])*(len(ring['alpha'])-1),
                                   0,len(ring['alpha'])-1).astype(np.int32)
                    fi = np.zeros_like(ndl); fi[inring] = ring['alpha'][xidx[inring]]
                    ndl = ndl*(1-0.88*fi)
            na = nx*axis[0]+ny*axis[1]+nz*axis[2]
            n1_ = nx*e1[0]+ny*e1[1]+nz*e1[2]; n2_ = nx*e2[0]+ny*e2[1]+nz*e2[2]
            lat = np.arcsin(np.clip(na,-1,1)); lon = np.arctan2(n2_,n1_)+lon_t
            uu = (lon/(2*math.pi)+0.5)*htw; vv = (0.5-lat/math.pi)*hth
            col = sample_eirect = sample_equirect(tex,uu,vv)
            mu = np.clip(-(nx*dx+ny*dy+nz*dz),0,1)
            ld = 1-b.limbdark*(1-mu)
            light = ndl*ld
            out = col*(0.045+light*1.42*b.gain)[...,None]
            cl = None
            if b.clouds is not None:
                cl = np.clip(sample_equirect(b.clouds,uu,vv).mean(-1)*0.62,0,0.62)
                out = out*(1-cl[...,None])+np.array([1.02,1.0,0.98])[None,None,:]*(cl*ndl*1.05)[...,None]
            if b.night is not None:
                nt = sample_equirect(b.night,uu,vv)
                lights = nt.mean(-1)**1.4
                night_amt = np.clip((0.05-ndl)*12,0,1)
                if cl is not None: night_amt *= np.clip(1-cl,0,1)
                out = out+lights[...,None]*np.array([1.0,0.75,0.45])[None,None,:]*night_amt[...,None]*0.9
            if b.atmo is not None:
                fres = (1-mu)**3.2
                out = out+np.array(b.atmo['col'],np.float32)[None,None,:]*(fres*(ndl*0.75+0.06)*b.atmo['k'])[...,None]
            sub = img[jj0:jj1,i0:i1]
            sub[hit] = out[hit]

    def draw_ring(s,img,E,f,r,u,tanf,b,t,chunk=150):
        ring = b.ring; C = b.center(t)
        inner,outer = ring['inner'],ring['outer']
        rel = C-E
        cz = max(rel@f,1e-3)
        rad_px = math.tan(math.atan(outer/cz))/tanf*(H/2)
        cx = ((rel@r)/cz)/tanf*(H/2)+W/2
        cy = -((rel@u)/cz)/tanf*(H/2)+H/2
        pad = rad_px*1.15+6
        i0,i1 = max(int(cx-pad),0),min(int(cx+pad),W)
        j0,j1 = max(int(cy-pad),0),min(int(cy+pad),H)
        if i1<=i0 or j1<=j0: return
        m = b.axis; R = b.radius
        prof,pc = (ring['alpha'],ring['color']) if not ring.get('thin') else (None,None)
        oc = E-C
        for jj0 in range(j0,j1,chunk):
            jj1 = min(jj0+chunk,j1)
            ys,xs = np.mgrid[jj0:jj1,i0:i1]
            pxs = ((xs+0.5-W/2)/(H/2))*tanf
            pys = ((H/2-(ys+0.5))/(H/2))*tanf
            dx = f[0]+r[0]*pxs+u[0]*pys; dy = f[1]+r[1]*pxs+u[1]*pys; dz = f[2]+r[2]*pxs+u[2]*pys
            dl = 1.0/np.sqrt(dx*dx+dy*dy+dz*dz); dx*=dl; dy*=dl; dz*=dl
            den = dx*m[0]+dy*m[1]+dz*m[2]
            num = (C-E)@m
            okd = np.abs(den)>1e-7
            tq = np.where(okd,num/np.where(okd,den,1),-1)
            OK = tq>0
            Px = E[0]+dx*tq; Py = E[1]+dy*tq; Pz = E[2]+dz*tq
            qx,qy,qz = Px-C[0],Py-C[1],Pz-C[2]
            rho = np.sqrt(qx*qx+qy*qy+qz*qz)
            inring = OK&(rho>inner)&(rho<outer)
            if not inring.any(): continue
            bq = oc[0]*dx+oc[1]*dy+oc[2]*dz; cq = oc@oc-R*R   # 行星遮挡
            disc = bq*bq-cq; blk = disc>0
            t_pl = -bq-np.sqrt(np.where(blk,disc,0))
            blk &= (t_pl>0)&(t_pl<tq)
            pln = 1.0/np.sqrt(Px*Px+Py*Py+Pz*Pz)               # 行星影子
            uhx,uhy,uhz = Px*pln,Py*pln,Pz*pln
            proj = C[0]*uhx+C[1]*uhy+C[2]*uhz
            ddx,ddy,ddz = C[0]-proj*uhx,C[1]-proj*uhy,C[2]-proj*uhz
            hdist = np.sqrt(ddx*ddx+ddy*ddy+ddz*ddz)
            shadow = (proj>0)&(hdist<R*1.03)
            xr = (rho-inner)/(outer-inner)
            if prof is not None:
                xidx = np.clip(xr*(len(prof)-1),0,len(prof)-1)
                xi = xidx.astype(np.int32); xf = xidx-xi
                a = prof[xi]*(1-xf)+prof[np.minimum(xi+1,len(prof)-1)]*xf
                cc = pc[xi]*(1-xf[...,None])+pc[np.minimum(xi+1,len(prof)-1)]*xf[...,None]
                a = np.where(shadow,a*0.12,a)
            else:
                a = 0.36*np.exp(-((xr-0.5)/0.07)**2)+0.10*np.exp(-((xr-0.15)/0.05)**2)+0.08*np.exp(-((xr-0.85)/0.05)**2)
                a = np.where(shadow,a*0.15,a).astype(np.float32)
                cc = np.array([0.62,0.58,0.55],np.float32)[None,None,:]*np.ones(a.shape+(1,),np.float32)
            a = np.where(blk,0.0,a)
            vis = inring&(a>0.004)
            if not vis.any(): continue
            sunh = np.clip((qx*(-uhx)+qy*(-uhy)+qz*(-uhz))/np.maximum(rho,1e-6),0,1)
            rcol = cc*(0.16+1.15*sunh)[...,None]
            sub = img[jj0:jj1,i0:i1]
            al = a[...,None]
            sub[vis] = sub[vis]*(1-al[vis])+rcol[vis]*al[vis]

    def draw_sun(s,img,E,f,r,u,tanf,b,t):
        C = np.zeros(3); R = b.radius
        rel = C-E; dist = np.linalg.norm(rel)
        if dist<=R*1.02: return
        cz = rel@f
        if cz<0.05: return
        rad_px = math.tan(math.asin(min(0.999,R/dist)))/tanf*(H/2)
        if rad_px>3:
            s.draw_sphere_emissive(img,E,f,r,u,tanf,b,t)
        cx = ((rel@r)/cz)/tanf*(H/2)+W/2
        cy = -((rel@u)/cz)/tanf*(H/2)+H/2
        if not(0<=cx<W and 0<=cy<H): return
        cor = 2.4 if rad_px>30 else 2.0
        bb = rad_px*cor+30
        i0,i1 = max(int(cx-bb),0),min(int(cx+bb),W)
        j0,j1 = max(int(cy-bb),0),min(int(cy+bb),H)
        if i1>i0 and j1>j0:
            ys,xs = np.mgrid[j0:j1,i0:i1]
            dpx = np.sqrt((xs-cx)**2+(ys-cy)**2)+1e-3
            ratio = rad_px/dpx
            g = np.clip(ratio**4.0,0,1)*0.55+np.clip(ratio**2.0,0,1)*0.055
            g[dpx<rad_px*0.995] = 0
            edge = np.clip((bb-dpx)/(0.35*bb),0,1)      # 盒边平滑归零，消除方形边界
            g *= edge*edge*(3-2*edge)
            img[j0:j1,i0:i1] += g[...,None]*np.array([1.0,0.88,0.68],np.float32)[None,None,:]
        if rad_px<80:                       # 远处：点光源刺芒（物理上仍是极亮星）
            wc = np.array([1.0,0.97,0.90],np.float32)
            L = int(min(500,90+14000/(dist+8)))
            xs = np.arange(-L,L+1)
            prof = np.exp(-(xs/(L*0.20))**2)*0.85
            x0f,x1f = max(int(cx)-L,0),min(int(cx)+L+1,W)
            if x1f>x0f:
                seg = prof[x0f-(int(cx)-L):x1f-(int(cx)-L)]
                img[int(cy),x0f:x1f] += wc[None,:]*seg[:,None]
                img[int(cy)-1,x0f:x1f] += wc[None,:]*seg[:,None]*0.4
                img[int(cy)+1,x0f:x1f] += wc[None,:]*seg[:,None]*0.4
            Lv = L//2
            ys2 = np.arange(-Lv,Lv+1)
            prof2 = np.exp(-(ys2/(Lv*0.14))**2)*0.45
            y0f,y1f = max(int(cy)-Lv,0),min(int(cy)+Lv+1,H)
            if y1f>y0f:
                seg2 = prof2[y0f-(int(cy)-Lv):y1f-(int(cy)-Lv)]
                img[y0f:y1f,int(cx)] += wc[None,:]*seg2[:,None]
            i0,i1 = max(int(cx)-40,0),min(int(cx)+41,W); j0,j1 = max(int(cy)-40,0),min(int(cy)+41,H)
            ys3,xs3 = np.mgrid[j0:j1,i0:i1]
            dd = np.sqrt((xs3-cx)**2+(ys3-cy)**2)+1e-3
            g2 = np.exp(-(dd/8.0)**2)*2.2+np.exp(-(dd/22.0)**2)*0.5
            img[j0:j1,i0:i1] += g2[...,None]*np.array([1.0,0.95,0.85],np.float32)[None,None,:]

    def draw_sphere_emissive(s,img,E,f,r,u,tanf,b,t,chunk=150):
        C = b.center(t); R = b.radius
        rel = C-E
        cz = rel@f
        if cz<0.05: return
        rad_px = math.tan(math.asin(min(1.0,R/np.linalg.norm(rel))))/tanf*(H/2)
        cx = ((rel@r)/cz)/tanf*(H/2)+W/2
        cy = -((rel@u)/cz)/tanf*(H/2)+H/2
        pad = rad_px*1.05+4
        i0,i1 = max(int(cx-pad),0),min(int(cx+pad),W)
        j0,j1 = max(int(cy-pad),0),min(int(cy+pad),H)
        if i1<=i0 or j1<=j0: return
        oc = E-C; tex = b.tex; htw,hth = tex.shape[1],tex.shape[0]
        axis,e1,e2 = b.axis,b.e1,b.e2; lon_t = b.lon0+b.rot*t
        for jj0 in range(j0,j1,chunk):
            jj1 = min(jj0+chunk,j1)
            ys,xs = np.mgrid[jj0:jj1,i0:i1]
            pxs = ((xs+0.5-W/2)/(H/2))*tanf
            pys = ((H/2-(ys+0.5))/(H/2))*tanf
            dx = f[0]+r[0]*pxs+u[0]*pys; dy = f[1]+r[1]*pxs+u[1]*pys; dz = f[2]+r[2]*pxs+u[2]*pys
            dl = 1.0/np.sqrt(dx*dx+dy*dy+dz*dz); dx*=dl; dy*=dl; dz*=dl
            bq = oc[0]*dx+oc[1]*dy+oc[2]*dz; cq = oc@oc-R*R
            disc = bq*bq-cq; hit = disc>0
            if not hit.any(): continue
            tq = -bq-np.sqrt(np.where(hit,disc,0)); hit &= tq>0
            Px = E[0]+dx*tq; Py = E[1]+dy*tq; Pz = E[2]+dz*tq
            nx,ny,nz = (Px-C[0])/R,(Py-C[1])/R,(Pz-C[2])/R
            mu = np.clip(-(nx*dx+ny*dy+nz*dz),0,1)
            na = nx*axis[0]+ny*axis[1]+nz*axis[2]
            n1_ = nx*e1[0]+ny*e1[1]+nz*e1[2]; n2_ = nx*e2[0]+ny*e2[1]+nz*e2[2]
            lat = np.arcsin(np.clip(na,-1,1)); lon = np.arctan2(n2_,n1_)+lon_t
            uu = (lon/(2*math.pi)+0.5)*htw; vv = (0.5-lat/math.pi)*hth
            col = sample_eirect_e = sample_equirect(tex,uu,vv)
            # 光球近似均匀：纹理仅保留35%（真实太阳颗粒对比~15%，γ2.2线性化会放大暗斑）
            col = col*0.35+0.65*np.array([1.0,0.62,0.24],np.float32)[None,None,:]
            out = col*(2.3*(0.25+0.75*mu**0.60))[...,None]
            sub = img[jj0:jj1,i0:i1]
            sub[hit] = out[hit]

# ---------------------------------------------------------------- 字幕
def load_fonts():
    def pick(sz):
        for p in ['/System/Library/Fonts/Hiragino Sans GB.ttc',
                  '/System/Library/Fonts/STHeiti Medium.ttc',
                  '/System/Library/Fonts/STHeiti Light.ttc']:
            if os.path.exists(p):
                try: return ImageFont.truetype(p,sz)
                except Exception: pass
        return ImageFont.load_default()
    return dict(title=pick(84),sub=pick(34),name=pick(46),fact=pick(27),
                hud=pick(25),end=pick(52),credit=pick(24))

CAPTIONS = [
    (0.7,3.8,('太阳系巡航','SOLAR SYSTEM · GRAND TOUR','天体尺寸按真实比例 · 轨道距离与时间经压缩'),'title'),
    (4.2,7.4,('太阳 SUN','直径139万公里 · 占太阳系总质量99.86%','阳光到达地球需8分20秒'),'name'),
    (8.4,13.8,('水星 MERCURY','0.39 AU · 直径4,879 km','无大气，昼夜温差超过600°C'),'name'),
    (17.0,21.6,('金星 VENUS','0.72 AU · 自转一周243天，比它的公转更慢','表面464°C · 顶层为硫酸云'),'name'),
    (24.2,30.6,('地球 & 月球 EARTH & MOON','1 AU · 直径12,742 km · 唯一已知生命世界','月球直径3,474 km（0.27倍地球）· 距地38万公里','这段距离足以并排放下太阳系其余全部行星'),'name'),
    (32.8,37.2,('火星 MARS','1.52 AU · 直径6,779 km','奥林帕斯山高约22 km，太阳系最高火山'),'name'),
    (43.0,50.4,('木星 JUPITER','5.2 AU · 质量是其余行星总和的2.5倍','大红斑：被持续观测350余年的巨型风暴'),'name'),
    (52.8,58.6,('土星 SATURN','9.5 AU · 星环主体为水冰碎块','环系跨度28万公里，平均厚度仅约数十米'),'name'),
    (61.2,65.6,('天王星 URANUS','19.2 AU · 自转轴倾斜98°','“躺着”公转 · 大气低至-224°C'),'name'),
    (68.0,72.2,('海王星 NEPTUNE','30.1 AU · 风速达2,100 km/h，太阳系之最','1846年先由数学预言、后经观测证实'),'name'),
    (74.8,79.4,('冥王星 & 卡戎 PLUTO & CHARON','平均39.5 AU · 相互潮汐锁定，质心在冥王星之外','“心形”汤博区为氮冰平原 · 卡戎直径1,212 km（约冥王星一半）'),'name'),
    (80.6,85.2,('赛德娜 SEDNA','近日点76 AU · 远日点约937 AU','公转一周约11,400年 · 表面异常深红','（尚无清晰影像，此为基于反照率的再现）'),'name'),
    (87.5,92.4,('眺望比邻星方向 · TOWARD PROXIMA','4.24光年外的M5.5红矮星，直径约为太阳的1/7','太暗，肉眼不可见——它就藏在这片星海之中','已确认位于宜居带的行星“比邻星b”'),'name'),
]
END_CARD = ('下一站 · 半人马座 α',
            '尺寸比例真实 · 轨道距离压缩 · 程序化逐帧渲染',
            '纹理 NASA / JPL-Caltech / USGS / JHU APL·SWRI / SolarSystemScope (CC BY 4.0)')

def _ctext(d,cx,y,s,font,col,shadow=2):
    if shadow: d.text((cx+shadow,y+shadow),s,font=font,fill=(0,0,0,min(160,col[-1])),anchor='ma')
    d.text((cx,y),s,font=font,fill=col,anchor='ma')
def _ltext(d,x,y,s,font,col,shadow=2):
    if shadow: d.text((x+shadow,y+shadow),s,font=font,fill=(0,0,0,min(160,col[-1])),anchor='la')
    d.text((x,y),s,font=font,fill=col,anchor='la')

def draw_captions(pil,t,F,prox_xy=None):
    d = ImageDraw.Draw(pil,'RGBA')
    fade = 0.6
    if prox_xy is not None and 87.5 <= t <= 92.8:
        a = int(235*min(1.0,(t-87.5)/1.0, max(0.0,(92.8-t))/0.6))
        x,y = prox_xy; rr = 30
        col = (255,208,150,a)
        d.ellipse([x-rr,y-rr,x+rr,y+rr],outline=col,width=2)
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
            d.line([x+dx*(rr+4),y+dy*(rr+4),x+dx*(rr+14),y+dy*(rr+14)],fill=col,width=2)
        lx,ly = (x+rr+26,y+rr+10) if x < W-560 else (x-rr-460,y+rr+10)
        d.line([x+rr*0.7,y+rr*0.7,lx-8,ly+34],fill=col,width=1)
        _ltext(d,lx,ly,'比邻星 Proxima',F['fact'],(255,215,165,a))
        _ltext(d,lx,ly+34,'4.24 光年 · 星等11 · 肉眼不可见',F['credit'],(220,185,150,int(a*0.85)))
    for (t0,t1,lines,kind) in CAPTIONS:
        if not(t0<=t<=t1): continue
        a = int(255*min(1.0,(t-t0)/fade,(t1-t)/fade))
        if kind=='title':
            cx = W//2; y0 = int(H*0.30)
            _ctext(d,cx,y0,lines[0],F['title'],(255,255,255,a),3)
            _ctext(d,cx,y0+82,lines[1],F['sub'],(200,210,235,int(a*0.88)))
            _ctext(d,cx,y0+130,lines[2],F['credit'],(160,170,190,int(a*0.75)))
        else:
            y = H-206
            _ltext(d,90,y,lines[0],F['name'],(240,244,255,a))
            for i,ln in enumerate(lines[1:]):
                _ltext(d,92,y+64+i*38,ln,F['fact'],(185,195,215,int(a*0.86)))
    if t>92.8:
        a = min(1.0,(t-92.8)/0.8)
        if t>95.2: a = min(a,max(0.0,(96.0-t)/0.8))
        ov = Image.new('RGBA',(W,H),(2,4,10,int(150*a)))
        pil.paste(Image.alpha_composite(pil.convert('RGBA'),ov).convert('RGB'),(0,0))
        d = ImageDraw.Draw(pil,'RGBA')
        _ctext(d,W//2,int(H*0.42),END_CARD[0],F['end'],(245,248,255,int(255*a)),3)
        _ctext(d,W//2,int(H*0.42)+84,END_CARD[1],F['fact'],(190,200,220,int(220*a)))
        _ctext(d,W//2,int(H*0.42)+126,END_CARD[2],F['credit'],(150,160,180,int(200*a)))
    if 4.0<t<88.0:
        au = float(np.interp(t,AU_KEY_T,AU_KEY_V))
        txt = f'日心距离 ≈ {au*149.6:.0f} 万公里' if au<0.1 else f'日心距离 ≈ {au:.2f} AU'
        tw = d.textlength(txt,font=F['hud'])
        _ltext(d,W-90-tw,H-64,txt,F['hud'],(170,180,200,200))
    elif 88.0<=t<=92.8:
        txt = '航向：半人马座α · 4.24 光年'
        tw = d.textlength(txt,font=F['hud'])
        _ltext(d,W-90-tw,H-64,txt,F['hud'],(255,175,135,225))

# ---------------------------------------------------------------- 主流程
G = {}
def worker_init():
    global G
    os.environ['OMP_NUM_THREADS']='1'
    G['R'] = Renderer()
    G['F'] = load_fonts()

def render_frame(fi):
    t = old_time(fi/FPS)
    pil = Image.fromarray(G['R'].render(t))
    px = None
    if 87.5 <= t <= 92.8:
        E,f,r,u,tanf = G['R'].camera(t)
        zc = float(G['R'].prox@f)
        if zc > 0.3:
            sx = (G['R'].prox@r/zc)/tanf*(H/2)+W/2
            sy = -(G['R'].prox@u/zc)/tanf*(H/2)+H/2
            if 120 < sx < W-120 and 120 < sy < H-160: px = (sx,sy)
    draw_captions(pil,t,G['F'],px)
    return pil

def do_frame(fi):
    render_frame(fi).save(os.path.join(OUTF,f'{fi:05d}.jpg'),quality=93)

def cmd_preview(times):
    os.makedirs(PREV,exist_ok=True)
    worker_init()
    for t in times:
        fi = int(t*FPS); st = time.time()
        render_frame(fi).save(os.path.join(PREV,f'pv_{t:05.1f}.jpg'),quality=90)
        print(f't={t:5.1f}s  {time.time()-st:.2f}s/帧',flush=True)

def cmd_render(a=0,b=NFRAMES,procs=6):
    os.makedirs(OUTF,exist_ok=True)
    import multiprocessing as mp
    todo = [i for i in range(a,b) if not os.path.exists(os.path.join(OUTF,f'{i:05d}.jpg'))]
    if not todo: print('全部帧已存在'); return
    print(f'渲染 {len(todo)} 帧 / {procs} 进程')
    t0 = time.time()
    with mp.get_context('fork').Pool(procs,initializer=worker_init) as pool:
        for n,_ in enumerate(pool.imap_unordered(do_frame,todo,chunksize=8),1):
            if n%100==0:
                el = time.time()-t0
                print(f'{n}/{len(todo)}  已用{el:.0f}s  预计剩余{el/n*(len(todo)-n):.0f}s',flush=True)
    print(f'完成，用时 {time.time()-t0:.0f}s')

def cmd_audio(path=None):
    path = path or os.path.join(VID,'tour_audio.wav')
    sr = 44100; T = DUR_OUT            # 直接在输出时间轴合成，无重采样
    n = int(sr*T); t = np.arange(n)/sr
    rng = np.random.default_rng(5)
    def voice(freq,tt,det=0.0016):
        lfo = 1+0.006*np.sin(2*np.pi*0.11*tt+freq*0.01)
        v = np.sin(2*np.pi*freq*lfo*tt)+0.35*np.sin(2*np.pi*freq*2.003*tt+1.3)+0.12*np.sin(2*np.pi*freq*3.01*tt+0.4)
        return 0.6*v+0.4*np.sin(2*np.pi*freq*(1+det)*tt)
    chords = [(0,16,[110.0,164.81,220.0]),(14,28,[87.31,130.81,174.61]),
              (26,40,[130.81,196.0,329.63]),(38,56,[98.0,146.83,196.0,246.94]),
              (54,66,[82.41,123.47,164.81]),(64,78,[110.0,164.81,246.94,261.63]),
              (76,96,[146.83,220.0,293.66,369.99])]
    mix = np.zeros(n)
    for (c0,c1,fs) in chords:
        n0,n1 = float(c0),float(c1)
        i0,i1 = int(n0*sr),min(int(n1*sr),n)
        seg = t[i0:i1]
        env = np.clip(np.minimum((seg-n0)/3.0,(n1-seg)/3.0),0,1)**1.5
        acc = np.zeros(i1-i0)
        for f_ in fs: acc += voice(f_,seg)
        mix[i0:i1] += acc*env*0.16
    mix += 0.10*np.sin(2*np.pi*55*t)+0.08*np.sin(2*np.pi*55.45*t)     # 引擎低鸣
    br = np.cumsum(rng.normal(0,1,n)); br/= (np.abs(br).max()+1e-9)
    lp = np.convolve(br,np.ones(64)/64,mode='same')
    mix += lp*0.05
    for tc in [10.2,18.8,25.9,27.6,34.2,46.4,55.0,62.8,69.6,76.2,83.8]:
        mix += lp*np.exp(-((t-tc)/0.7)**2)*0.14         # 交会拂过
    mix *= np.clip(np.minimum(t/1.5,(T-t)/2.5),0,1)
    delay = int(0.0006*sr)
    st = np.stack([mix,np.concatenate([mix[delay:],mix[:delay]])],-1)
    st = st/(np.abs(st).max()+1e-9)*0.55
    with wave.open(path,'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((st*32767).astype('<i2').tobytes())
    print('audio ->',path)

def cmd_encode(out=None):
    out = out or os.path.join(VID,'solar_system_tour.mp4')
    aud = os.path.join(VID,'tour_audio.wav')
    if not os.path.exists(aud): cmd_audio(aud)
    subprocess.run([FFMPEG,'-y','-framerate',str(FPS),'-i',os.path.join(OUTF,'%05d.jpg'),
                    '-i',aud,'-c:v','libx264','-crf','18','-preset','slow','-pix_fmt','yuv420p',
                    '-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',out],check=True)
    print('->',out)

def cmd_sky():
    os.makedirs(PREV,exist_ok=True)
    mw,st_,px,aA,aB = build_sky()
    Image.fromarray((aces(mw*3.2)*255).astype(np.uint8)).save(os.path.join(PREV,'sky_mw.png'))
    print('sky ok; stars:',st_[0].shape)

if __name__=='__main__':
    cmd = sys.argv[1] if len(sys.argv)>1 else 'preview'
    if cmd=='sky': cmd_sky()
    elif cmd=='preview': cmd_preview([float(x) for x in sys.argv[2:]])
    elif cmd=='render':
        a = int(sys.argv[2]) if len(sys.argv)>2 else 0
        b = int(sys.argv[3]) if len(sys.argv)>3 else NFRAMES
        cmd_render(a,b)
    elif cmd=='audio': cmd_audio()
    elif cmd=='encode': cmd_encode()
