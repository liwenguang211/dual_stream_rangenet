"""
Dual-Stream RangeNet Lite v2 — 网络结构可视化
Stream-1: Range+Intensity (2ch)
Stream-2: Normal+XYZ     (6ch)
Fusion:   CBAM (Channel+Spatial Attention)
Backbone: DSConv U-Net
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# ─────────────────────────────────────────────────────────────
# 全局字体 (Noto Sans CJK)
# ─────────────────────────────────────────────────────────────
_CJK_FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
_font_prop = fm.FontProperties(fname=_CJK_FONT, size=9)
_cjk_name  = fm.FontProperties(fname=_CJK_FONT).get_name()
fm.fontManager.ttflist += fm.createFontList([_CJK_FONT])

plt.rcParams.update({
    'font.family': _cjk_name,
    'font.size':    9,
    'axes.linewidth': 0,
})

# ─────────────────────────────────────────────────────────────
# 配色方案
# ─────────────────────────────────────────────────────────────
C = {
    'bg':       '#F0F4F8',
    'input':    '#2C3E50',
    's1_stem':  '#154360',   # 深蓝  — Stream-1 stem
    's1_enc':   '#1F618D',   # 蓝    — Stream-1 encoder
    's2_stem':  '#4A235A',   # 深紫  — Stream-2 stem
    's2_enc':   '#7D3C98',   # 紫    — Stream-2 encoder
    'aspp':     '#117A65',   # 深青绿 — ASPP
    'cbam':     '#D35400',   # 橙    — CBAM Fusion
    'ch_attn':  '#E67E22',   # 浅橙  — Channel Attention
    'sp_attn':  '#CA6F1E',   # 深橙  — Spatial Attention
    'decoder':  '#1E8449',   # 绿    — Decoder
    'head':     '#922B21',   # 红    — Head
    'output':   '#2C3E50',
    'res':      '#7F8C8D',   # 灰    — ResBlock badge
    'skip':     '#27AE60',   # 绿    — skip arrows
    'arrow':    '#4D5656',
}

FIG_W, FIG_H = 24, 15
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C['bg'])
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor(C['bg'])

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def box(x, y, w, h, color, alpha=0.92, radius=0.18, zorder=3):
    fc = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=color, edgecolor='white',
        linewidth=1.2, alpha=alpha, zorder=zorder)
    ax.add_patch(fc)

def label(x, y, text, size=8.2, color='white', bold=False, zorder=5):
    ax.text(x, y, text, ha='center', va='center',
            fontsize=size, color=color,
            fontweight='bold' if bold else 'normal', zorder=zorder)

def badge(x, y, text, color, size=6.5):
    ax.text(x, y, text, ha='center', va='center',
            fontsize=size, color='white',
            bbox=dict(boxstyle='round,pad=0.15', facecolor=color,
                      edgecolor='none', alpha=0.9),
            zorder=6)

def dim(x, y, text, size=6.8):
    ax.text(x, y, text, ha='center', va='center',
            fontsize=size, color='#555555', style='italic', zorder=6)

def arr(x1, y1, x2, y2, color='#4D5656', lw=1.4, mutation=12, zorder=4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=mutation),
                zorder=zorder, alpha=0.85)

def skip_arr(x1, y1, x2, y2, color, lw=1.05, rad=0.0, zorder=3):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle='dashed', mutation_scale=9,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=zorder, alpha=0.72)

def section_bg(x0, x1, y0, y1, color, alpha=0.07, txt='', lx=None, ly=None):
    r = mpatches.FancyBboxPatch(
        (x0, y0), x1-x0, y1-y0,
        boxstyle='round,pad=0.1,rounding_size=0.3',
        facecolor=color, edgecolor=color,
        linewidth=1.5, alpha=alpha, zorder=1)
    ax.add_patch(r)
    if txt:
        ax.text(lx or (x0+x1)/2, ly or y1-0.22, txt,
                ha='center', va='top', fontsize=8.2, color=color,
                fontweight='bold', alpha=0.75, zorder=2)

# ─────────────────────────────────────────────────────────────
# 布局坐标
# ─────────────────────────────────────────────────────────────
Y1      = 11.0   # Stream-1 中轴  (Range+Intensity)
Y2      =  4.5   # Stream-2 中轴  (Normal+XYZ)
Y_DEC   =  7.75  # Decoder / Fusion 中轴

X_IN1   =  1.3   # Stream-1 输入
X_IN2   =  1.3   # Stream-2 输入 (同x, 不同y)
X_STEM  =  3.0   # Stem
X_E1    =  4.85  # Enc-1  H/2
X_E2    =  6.7   # Enc-2  H/4
X_E3    =  8.55  # Enc-3  H/8  (+ASPP)
X_F3    = 10.5   # CBAM Fusion @ H/8
X_D2    = 12.3   # Up-2   H/4
X_D1    = 14.1   # Up-1   H/2
X_D0    = 15.9   # Up-0   H
X_HEAD  = 17.7   # Head
X_OUT   = 19.5   # Output 后处理列
X_CLS   = 22.0   # 类别图例

BW  = 1.3    # block width
BH  = 0.72   # block height
BHS = 0.62   # small block height

# ─────────────────────────────────────────────────────────────
# 背景分区
# ─────────────────────────────────────────────────────────────
section_bg( 1.9, 10.0,  9.5, 12.5, C['s1_enc'],
            txt='Stream-1: Range Stream  (Range + Intensity, 2ch)', lx=5.9, ly=12.45)
section_bg( 1.9, 10.0,  2.8,  5.8, C['s2_enc'],
            txt='Stream-2: Geometry Stream  (Normal nx,ny,nz + XYZ, 6ch)', lx=5.9, ly=5.75)
section_bg( 9.85, 11.2,  2.8, 12.5, C['cbam'],
            txt='CBAM\nFusion', lx=10.52, ly=12.45)
section_bg(11.2, 17.2,  6.3,  9.2, C['decoder'],
            txt='Decoder — DSConv U-Net', lx=14.2, ly=9.15)

# ─────────────────────────────────────────────────────────────
# 1. 输入框
# ─────────────────────────────────────────────────────────────
# Stream-1 输入
box(X_IN1, Y1, 1.15, 1.5, C['s1_stem'], alpha=0.9, radius=0.22)
label(X_IN1, Y1+0.45, 'Range Image', size=7.8, bold=True)
label(X_IN1, Y1+0.05, '[1, 2, H, W]', size=7.0)
label(X_IN1, Y1-0.3, 'range  intensity', size=6.2)
# 通道色条
for ci, (cc, cn) in enumerate(zip(['#E74C3C','#F5CBA7'], ['r','i'])):
    bx = X_IN1 - 0.12 + ci * 0.26
    r = mpatches.FancyBboxPatch((bx-0.10, Y1+0.65), 0.20, 0.48,
        boxstyle='round,pad=0,rounding_size=0.03',
        facecolor=cc, alpha=0.85, zorder=6, edgecolor='white', lw=0.5)
    ax.add_patch(r)
    ax.text(bx, Y1+0.90, cn, ha='center', va='center',
            fontsize=6.5, color='white', fontweight='bold', zorder=7)

# Stream-2 输入
box(X_IN2, Y2, 1.15, 1.6, C['s2_stem'], alpha=0.9, radius=0.22)
label(X_IN2, Y2+0.52, 'Geometry Map', size=7.8, bold=True)
label(X_IN2, Y2+0.10, '[1, 6, H, W]', size=7.0)
label(X_IN2, Y2-0.25, 'nx  ny  nz', size=6.0)
label(X_IN2, Y2-0.52, 'x    y    z', size=6.0)
# 通道色条
geo_cols = ['#2980B9','#1ABC9C','#27AE60','#8E44AD','#D35400','#E74C3C']
geo_names= ['nx','ny','nz','x','y','z']
for ci, (cc, cn) in enumerate(zip(geo_cols, geo_names)):
    bx = X_IN2 - 0.6 + ci * 0.24
    r = mpatches.FancyBboxPatch((bx-0.10, Y2+0.68), 0.20, 0.46,
        boxstyle='round,pad=0,rounding_size=0.03',
        facecolor=cc, alpha=0.85, zorder=6, edgecolor='white', lw=0.5)
    ax.add_patch(r)
    ax.text(bx, Y2+0.92, cn, ha='center', va='center',
            fontsize=5.5, color='white', fontweight='bold', zorder=7)

# 分流竖线 + 箭头
ax.plot([X_IN1+0.58, X_IN1+0.58], [Y2, Y1],
        color='#888888', lw=1.2, ls='-', zorder=4, alpha=0.55)
arr(X_IN1+0.58, Y1, X_STEM-BW/2-0.08, Y1, C['s1_enc'], lw=1.5)
arr(X_IN2+0.58, Y2, X_STEM-BW/2-0.08, Y2, C['s2_enc'], lw=1.5)

# ─────────────────────────────────────────────────────────────
# 2. Encoder 块构建函数
# ─────────────────────────────────────────────────────────────

def enc_block(x, y, title, sub, dims, color, w=BW, h=BH):
    box(x, y, w, h, color)
    label(x, y+0.19, title, size=8.2, bold=True)
    label(x, y-0.02, sub,   size=7.0)
    dim(x,  y-0.23, dims)

def enc_row(y, color_stem, color_enc, in_ch):
    """画一行编码器: Stem + Enc1 + Enc2 + Enc3+ASPP"""
    # Stem
    enc_block(X_STEM, y, 'Stem', f'Conv3×3+BN ({in_ch}→32)', '[32,H,W]', color_stem)
    badge(X_STEM, y-0.44, 'ResBlock', C['res'])

    # Enc-1
    enc_block(X_E1, y, 'Enc-1', 'DSConv s=2', '[64,H/2]', color_enc)
    badge(X_E1, y-0.44, 'ResBlock', C['res'])

    # Enc-2
    enc_block(X_E2, y, 'Enc-2', 'DSConv s=2', '[128,H/4]', color_enc)
    badge(X_E2, y-0.44, 'ResBlock', C['res'])

    # Enc-3 + ASPP
    enc_block(X_E3, y, 'Enc-3+ASPP', 'DSConv s=2', '[256,H/8]', C['aspp'])
    badge(X_E3, y-0.44, 'd=1,2,4', C['aspp'])

    # 箭头
    for x1, x2 in [(X_STEM,X_E1),(X_E1,X_E2),(X_E2,X_E3)]:
        arr(x1+BW/2, y, x2-BW/2-0.06, y, color_enc, lw=1.4)

# Stream-1
enc_row(Y1, C['s1_stem'], C['s1_enc'], in_ch=2)
# Stream-2
enc_row(Y2, C['s2_stem'], C['s2_enc'], in_ch=6)

# ─────────────────────────────────────────────────────────────
# 3. CBAM Fusion 块  (4 个尺度)
# ─────────────────────────────────────────────────────────────

def cbam_block(x, ch_out, dim_str):
    """单个 CBAM Fusion 竖块，居中于 Y_DEC"""
    yc = Y_DEC
    h  = 3.6
    box(x, yc, 0.9, h, C['cbam'], alpha=0.88, radius=0.2)
    label(x, yc+1.55, 'CBAM', size=8.5, bold=True)
    label(x, yc+1.20, 'Fusion', size=8.0, bold=True)
    # 内部步骤
    badge(x, yc+0.65, 'Concat', C['cbam'],       size=6.5)
    badge(x, yc+0.22, '1×1 Conv', C['cbam'],     size=6.5)
    # Channel Attn
    box(x, yc-0.28, 0.82, 0.44, C['ch_attn'], alpha=0.9, radius=0.12)
    label(x, yc-0.28, 'Ch-Attn', size=6.5)
    # Spatial Attn
    box(x, yc-0.80, 0.82, 0.44, C['sp_attn'], alpha=0.9, radius=0.12)
    label(x, yc-0.80, 'Sp-Attn', size=6.5)
    # Residual
    badge(x, yc-1.28, '+ Residual', C['cbam'],   size=6.2)
    dim(x,  yc-1.62, dim_str)

# Fusion@H/8  (连接 Enc-3/ASPP × 2)
cbam_block(X_F3, ch_out=256, dim_str='[256,H/8]')
# 从两个 ASPP → Fusion@H/8
arr(X_E3+BW/2, Y1, X_F3-0.45, Y1, C['aspp'],  lw=1.3)
arr(X_E3+BW/2, Y2, X_F3-0.45, Y2, C['aspp'],  lw=1.3)
# Fusion@H/8 → Decoder Up-2
arr(X_F3+0.45, Y_DEC, X_D2-BW/2-0.06, Y_DEC, C['cbam'], lw=1.5)

# Skip-level fusion boxes (F0/F1/F2) — small inline CBAM icons
SKIP_LABELS = [('F0','[32,H,W]'), ('F1','[64,H/2]'), ('F2','[128,H/4]')]
SKIP_XS = [X_STEM, X_E1, X_E2]
DEC_XS  = [X_D0,   X_D1,  X_D2]

for (fl, fdim), xs, xd in zip(SKIP_LABELS, SKIP_XS, DEC_XS):
    yf = 1.65   # 下方小融合块 y
    # 小 CBAM 融合块
    box(xs, yf, 0.82, 0.58, C['cbam'], alpha=0.80, radius=0.14, zorder=4)
    label(xs, yf+0.10, f'CBAM {fl}', size=7.0, bold=True)
    dim(xs,  yf-0.13, fdim)
    # Stream-1 enc → 小块
    skip_arr(xs, Y1-BH/2-0.05, xs, yf+0.29, C['s1_enc'], rad=0.0)
    # Stream-2 enc → 小块
    skip_arr(xs, Y2+BHS/2+0.05, xs, yf+0.29, C['s2_enc'], rad=0.0)
    # 小块 → Decoder
    ax.annotate('',
        xy=(xd, Y_DEC-BH/2-0.06), xytext=(xs, yf-0.29),
        arrowprops=dict(arrowstyle='->', color=C['skip'], lw=1.1,
                        linestyle='dashed', mutation_scale=9,
                        connectionstyle='arc3,rad=-0.15'),
        zorder=3, alpha=0.72)
    dim((xs+xd)/2, yf-0.92, fdim)

ax.text(5.5, 0.95, 'Skip CBAM (f0 / f1 / f2)',
        ha='center', va='center', fontsize=7.5,
        color=C['skip'], style='italic',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor=C['skip'], alpha=0.8), zorder=6)

# ─────────────────────────────────────────────────────────────
# 4. Decoder
# ─────────────────────────────────────────────────────────────

def dec_block(x, y, title, sub, dims, color=C['decoder'], w=BW, h=BH):
    box(x, y, w, h, color)
    label(x, y+0.19, title, size=8.2, bold=True)
    label(x, y-0.02, sub,   size=7.0)
    dim(x,  y-0.23, dims)

dec_block(X_D2, Y_DEC, 'Up-2', 'Bilinear+DSConv', '[128,H/4]')
badge(X_D2, Y_DEC-0.44, 'ResBlock', C['res'])

dec_block(X_D1, Y_DEC, 'Up-1', 'Bilinear+DSConv', '[64,H/2]')
badge(X_D1, Y_DEC-0.44, 'ResBlock', C['res'])

dec_block(X_D0, Y_DEC, 'Up-0', 'Bilinear+DSConv', '[32,H,W]')
badge(X_D0, Y_DEC-0.44, 'ResBlock', C['res'])

for x1, x2 in [(X_D2, X_D1), (X_D1, X_D0)]:
    arr(x1+BW/2, Y_DEC, x2-BW/2-0.06, Y_DEC, C['decoder'], lw=1.4)

# ─────────────────────────────────────────────────────────────
# 5. 分割头 → 输出 → 后处理
# ─────────────────────────────────────────────────────────────
arr(X_D0+BW/2, Y_DEC, X_HEAD-BW/2-0.06, Y_DEC, C['decoder'], lw=1.4)

box(X_HEAD, Y_DEC, BW, BH, C['head'], radius=0.18)
label(X_HEAD, Y_DEC+0.19, 'Head', size=8.5, bold=True)
label(X_HEAD, Y_DEC-0.02, 'DSConv+Conv1×1', size=7.0)
dim( X_HEAD, Y_DEC-0.23, '[9,H,W]')

arr(X_HEAD+BW/2, Y_DEC, X_OUT-0.42, Y_DEC, C['head'], lw=1.4)

# 后处理垂直列
pp_items = [
    (Y_DEC+0.9,  'Logits\n[1,9,H,W]',  C['head'],    0.8),
    (Y_DEC,      'Argmax',              '#7D3C98',    0.88),
    (Y_DEC-0.75, 'Back-\nProject',      '#117A65',    0.88),
    (Y_DEC-1.55, 'KNN\nFill',           '#1A5276',    0.88),
    (Y_DEC-2.35, 'Labels\n[N]',         C['output'],  0.9),
]
for (py, pt, pc, pa) in pp_items:
    box(X_OUT, py, 0.88, 0.58, pc, alpha=pa, radius=0.15)
    label(X_OUT, py, pt, size=7.2)

for y1, y2 in [(Y_DEC+0.61, Y_DEC+0.29),
               (Y_DEC-0.29, Y_DEC-0.46),
               (Y_DEC-1.04, Y_DEC-1.26),
               (Y_DEC-1.84, Y_DEC-2.06)]:
    arr(X_OUT, y1, X_OUT, y2, '#444444', lw=1.1)

# ─────────────────────────────────────────────────────────────
# 6. CBAM 内部说明框  (右上角)
# ─────────────────────────────────────────────────────────────
cx0, cy0 = 20.6, 14.0
box(cx0, cy0-1.1, 2.6, 2.15, C['cbam'], alpha=0.12, radius=0.3, zorder=2)
ax.text(cx0, cy0-0.08, 'CBAM Fusion 内部结构', ha='center', va='center',
        fontsize=8, color=C['cbam'], fontweight='bold', zorder=6)
steps = [
    ('Concat (r ++ g)',       C['cbam']),
    ('1×1 Conv + BN + ReLU', C['cbam']),
    ('Channel Attn (MLP)',    C['ch_attn']),
    ('Spatial Attn (Conv7)',  C['sp_attn']),
    ('+ Residual shortcut',   C['cbam']),
]
for si, (st, sc) in enumerate(steps):
    sy = cy0 - 0.45 - si * 0.34
    box(cx0, sy, 2.3, 0.28, sc, alpha=0.82, radius=0.1, zorder=5)
    label(cx0, sy, st, size=6.8, zorder=6)

# ─────────────────────────────────────────────────────────────
# 7. 类别图例
# ─────────────────────────────────────────────────────────────
cls_data = [
    (0, 'background',    '#111111'),
    (1, '地面 ground',   '#804080'),
    (2, '屋顶 roof',     '#464646'),
    (3, 'side_facade',   '#BE9999'),
    (4, 'front_facade',  '#9999BE'),
    (5, '横梁 beam',     '#FAAA1E'),
    (6, '柱子 column',   '#DCDC00'),
    (7, '窗户 window',   '#6B8E23'),
    (8, '动态 dynamic',  '#DC143C'),
]
cls_x = X_CLS
cy0_cls = 9.5
ax.text(cls_x, cy0_cls+0.3, '语义类别 (9类)', ha='center',
        fontsize=9, color='#333333', fontweight='bold')
for i, (cid, cname, ccolor) in enumerate(cls_data):
    cy = cy0_cls - 0.52 - i * 0.74
    rect = mpatches.FancyBboxPatch(
        (cls_x-0.78, cy-0.18), 0.44, 0.36,
        boxstyle='round,pad=0,rounding_size=0.05',
        facecolor=ccolor, edgecolor='white', lw=0.8,
        alpha=0.9, zorder=5)
    ax.add_patch(rect)
    ax.text(cls_x-0.56, cy, str(cid), ha='center', va='center',
            fontsize=7, color='white', fontweight='bold', zorder=6)
    ax.text(cls_x-0.28, cy, cname, ha='left', va='center',
            fontsize=7.5, color='#333333', zorder=6)

# ─────────────────────────────────────────────────────────────
# 8. 图例色块
# ─────────────────────────────────────────────────────────────
legend_items = [
    (C['s1_enc'],  'Stream-1 Range Encoder (2ch)'),
    (C['s2_enc'],  'Stream-2 Geometry Encoder (6ch)'),
    (C['aspp'],    'ASPP Bottleneck (d=1,2,4)'),
    (C['cbam'],    'CBAM Fusion (Ch-Attn + Sp-Attn)'),
    (C['decoder'], 'Decoder (Bilinear+DSConv)'),
    (C['head'],    'Segmentation Head (Conv1×1)'),
    (C['res'],     'ResBlock (DSConv residual)'),
    (C['skip'],    'Skip Connection (dashed)'),
]
lx0, ly0 = 1.1, 13.7
ax.text(lx0, ly0+0.15, '图例:', fontsize=8.5, color='#333333', fontweight='bold')
for i, (lc, lt) in enumerate(legend_items):
    col = i // 2
    row = i %  2
    lx = lx0 + col * 5.0
    ly = ly0 - 0.48 - row * 0.46
    rect = mpatches.FancyBboxPatch((lx, ly-0.14), 0.30, 0.28,
        boxstyle='round,pad=0,rounding_size=0.04',
        facecolor=lc, edgecolor='white', lw=0.6,
        alpha=0.92, zorder=5)
    ax.add_patch(rect)
    ax.text(lx+0.38, ly, lt, ha='left', va='center',
            fontsize=7.5, color='#333333', zorder=6)

# ─────────────────────────────────────────────────────────────
# 9. 标题与说明
# ─────────────────────────────────────────────────────────────
ax.text(FIG_W/2, 14.65,
        'Dual-Stream RangeNet Lite v2 — LiDAR 点云语义分割',
        ha='center', va='center', fontsize=15,
        color='#1A1A2E', fontweight='bold')
ax.text(FIG_W/2, 14.20,
        'Input: [1, 8, 64, 1024]  (range, intensity | nx, ny, nz, x, y, z)'
        '   →   Output: Semantic Labels [N points]  |  9 Classes  |  ~5.15M params',
        ha='center', va='center', fontsize=9, color='#555555')

# 标注轴方向
for x, txt in [(X_STEM,'H×W'), (X_E1,'H/2'), (X_E2,'H/4'), (X_E3,'H/8')]:
    ax.text(x, 9.02, txt, ha='center', fontsize=6.5,
            color='#888888', style='italic')
    ax.text(x, 6.30, txt, ha='center', fontsize=6.5,
            color='#888888', style='italic')

ax.text(X_F3,  6.30, 'H/8', ha='center', fontsize=6.5,
        color='#888888', style='italic')

# DSConv 注释
ax.text(1.15, 6.80,
        'DSConv =\nDepthwise Sep\nConv', ha='left', va='center',
        fontsize=6.8, color='#555555')

# ─────────────────────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────────────────────
from pathlib import Path

out = Path(__file__).resolve().parent / 'arch_diagram_v2.png'
plt.savefig(out, dpi=180, bbox_inches='tight',
            facecolor=C['bg'], edgecolor='none')
print(f'保存: {out}')
plt.close()
