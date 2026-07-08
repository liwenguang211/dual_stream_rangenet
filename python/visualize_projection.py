"""
visualize_projection.py — 球面投影可视化工具
================================================
功能：
  1. 球面投影 (x,y,z,intensity) → 2D 图像 (H=64, W=1024)，保留最近点（解决遮挡）
  2. 图像空间法向量计算：对 XYZ 图差分 → 叉乘 → 单位化
  3. 四窗口可视化：
       Stream 1: Depth (Range)       — JET 伪彩色深度图
       Stream 1: Intensity           — 灰度光强图
       Stream 2: Geometry (XYZ)      — RGB = (X, Y, Z) 归一化
       Stream 2: Direction Vectors   — RGB = (nx, ny, nz) 法向量图

用法：
  # 生成测试点云运行
  python visualize_projection.py

  # 读取 KITTI .bin 文件
  python visualize_projection.py --input scan.bin

  # 指定传感器 FOV（速腾/Velodyne）
  python visualize_projection.py --fov-up 15 --fov-down -15 --max-range 30
"""

import argparse
import sys
import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────
# 默认参数（与 model_config.h 保持一致）
# ─────────────────────────────────────────────────────────────
H           = 64
W           = 1024
FOV_UP_DEG  =  52.0    # Livox Mid-360 默认
FOV_DOWN_DEG = -7.0
MIN_RANGE   =   0.1
MAX_RANGE   =  40.0


# ─────────────────────────────────────────────────────────────
# 1. 球面投影
# ─────────────────────────────────────────────────────────────

def spherical_projection(pts: np.ndarray,
                         h: int, w: int,
                         fov_up_deg: float,
                         fov_down_deg: float,
                         min_range: float,
                         max_range: float):
    """
    将点云投影到 (h, w) 的球面图像上。

    参数
    ----
    pts : ndarray, shape (N, 4)  — [x, y, z, intensity]
    返回
    ----
    range_img  : (h, w)    float32  — 距离图（米）
    intensity_img : (h, w) float32  — 光强图（归一化）
    xyz_img    : (h, w, 3) float32  — XYZ 坐标图
    pixel_idx  : (h, w)    int32    — 每像素对应点索引（-1=空）
    """
    fov_up   = np.deg2rad(fov_up_deg)
    fov_down = np.deg2rad(fov_down_deg)
    fov      = abs(fov_up) + abs(fov_down)

    x, y, z, intensity = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    r = np.sqrt(x**2 + y**2 + z**2)

    # 过滤无效点
    valid = (r >= min_range) & (r <= max_range)
    x, y, z, intensity, r = x[valid], y[valid], z[valid], intensity[valid], r[valid]
    orig_idx = np.where(valid)[0]

    # 球面坐标
    yaw   = np.arctan2(y, x)                           # (-π, π)
    pitch = np.arcsin(np.clip(z / r, -1.0, 1.0))       # (-π/2, π/2)

    # 映射到像素坐标
    row = (1.0 - (pitch - fov_down) / fov) * h
    col = (0.5 * (1.0 - yaw / np.pi)) * w

    row = np.clip(row, 0, h - 1).astype(np.int32)
    col = np.clip(col, 0, w - 1).astype(np.int32)

    # 初始化图像
    range_img     = np.zeros((h, w), dtype=np.float32)
    intensity_img = np.zeros((h, w), dtype=np.float32)
    xyz_img       = np.zeros((h, w, 3), dtype=np.float32)
    pixel_idx     = np.full((h, w), -1, dtype=np.int32)
    depth_map     = np.full((h, w), np.inf, dtype=np.float32)

    # 按距离从远到近填充，保留最近点（解决遮挡）
    order = np.argsort(r)[::-1]   # 远 → 近，近的后写覆盖
    for i in order:
        pi, pj = row[i], col[i]
        if r[i] < depth_map[pi, pj]:
            depth_map[pi, pj]     = r[i]
            range_img[pi, pj]     = r[i]
            intensity_img[pi, pj] = intensity[i]
            xyz_img[pi, pj, 0]    = x[i]
            xyz_img[pi, pj, 1]    = y[i]
            xyz_img[pi, pj, 2]    = z[i]
            pixel_idx[pi, pj]     = orig_idx[i]

    return range_img, intensity_img, xyz_img, pixel_idx


# ─────────────────────────────────────────────────────────────
# 2. 图像空间法向量计算
# ─────────────────────────────────────────────────────────────

def compute_normals(xyz_img: np.ndarray,
                    range_img: np.ndarray) -> np.ndarray:
    """
    在图像空间利用有限差分 + 叉乘计算法向量。

    参数
    ----
    xyz_img   : (H, W, 3) — XYZ 坐标图
    range_img : (H, W)    — 距离图（用于有效像素掩码）

    返回
    ----
    normal_img : (H, W, 3) float32 — 单位法向量（nz, ny, nz）
    """
    h, w = range_img.shape
    valid = range_img > 0  # 有效像素掩码

    normal_img = np.zeros((h, w, 3), dtype=np.float32)

    # 上下/左右邻居坐标（有限差分）
    # dP/dr = P(r+1,c) - P(r-1,c)    垂直梯度
    # dP/dc = P(r,c+1) - P(r,c-1)    水平梯度
    xu = np.roll(xyz_img, -1, axis=0)   # 上移一行
    xd = np.roll(xyz_img,  1, axis=0)   # 下移一行
    xl = np.roll(xyz_img, -1, axis=1)   # 左移一列
    xr = np.roll(xyz_img,  1, axis=1)   # 右移一列

    vu = np.roll(valid, -1, axis=0)
    vd = np.roll(valid,  1, axis=0)
    vl = np.roll(valid, -1, axis=1)
    vr = np.roll(valid,  1, axis=1)

    # 四邻居均有效才计算
    valid_mask = valid & vu & vd & vl & vr

    # 梯度向量
    dr = xd - xu   # (H, W, 3) 垂直方向梯度
    dc = xr - xl   # (H, W, 3) 水平方向梯度

    # 叉积 N = dr × dc
    nx = dr[..., 1] * dc[..., 2] - dr[..., 2] * dc[..., 1]
    ny = dr[..., 2] * dc[..., 0] - dr[..., 0] * dc[..., 2]
    nz = dr[..., 0] * dc[..., 1] - dr[..., 1] * dc[..., 0]

    # 归一化
    length = np.sqrt(nx**2 + ny**2 + nz**2)
    length = np.where(length < 1e-6, 1.0, length)   # 避免除零

    normal_img[..., 0] = np.where(valid_mask, nx / length, 0)
    normal_img[..., 1] = np.where(valid_mask, ny / length, 0)
    normal_img[..., 2] = np.where(valid_mask, nz / length, 0)

    return normal_img


# ─────────────────────────────────────────────────────────────
# 3. 可视化辅助函数
# ─────────────────────────────────────────────────────────────

def _normalize_to_uint8(img: np.ndarray,
                        vmin: float = None,
                        vmax: float = None,
                        mask: np.ndarray = None) -> np.ndarray:
    """浮点图像归一化到 [0, 255] uint8，可选有效区域掩码。"""
    if mask is not None:
        vals = img[mask]
        if vals.size == 0:
            return np.zeros_like(img, dtype=np.uint8)
        lo = vals.min() if vmin is None else vmin
        hi = vals.max() if vmax is None else vmax
    else:
        lo = img.min() if vmin is None else vmin
        hi = img.max() if vmax is None else vmax

    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)

    out = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    if mask is not None:
        out[~mask] = 0
    return out


def make_depth_vis(range_img: np.ndarray) -> np.ndarray:
    """
    深度图 → JET 伪彩色 BGR 图像。
    近处偏蓝，远处偏红（COLORMAP_JET 特性）。
    """
    mask = range_img > 0
    gray = _normalize_to_uint8(range_img, vmin=MIN_RANGE, vmax=MAX_RANGE, mask=mask)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    color[~mask] = 0   # 空像素置黑
    return color


def make_intensity_vis(intensity_img: np.ndarray) -> np.ndarray:
    """光强图 → 灰度 BGR 图像（3 通道）。"""
    mask = intensity_img > 0
    gray = _normalize_to_uint8(intensity_img, vmin=0.0, vmax=1.0)
    bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr[~mask] = 0
    return bgr


def make_xyz_vis(xyz_img: np.ndarray, range_img: np.ndarray) -> np.ndarray:
    """
    XYZ 坐标图 → RGB 图像（R=X, G=Y, B=Z）。
    各通道独立归一化到 [0, 255]。
    """
    mask = range_img > 0
    out  = np.zeros((*xyz_img.shape[:2], 3), dtype=np.uint8)
    for c in range(3):
        out[..., c] = _normalize_to_uint8(xyz_img[..., c], mask=mask)
    # OpenCV 使用 BGR，XYZ → BGR = (Z, Y, X)
    bgr = out[..., ::-1].copy()
    bgr[~mask] = 0
    return bgr


def make_normal_vis(normal_img: np.ndarray) -> np.ndarray:
    """
    法向量图 → RGB 图像。
    法向量分量在 [-1, 1]，映射到 [0, 255]：0.5 对应中性灰。
    无法向量的像素（全零）置黑。
    """
    valid = np.any(normal_img != 0, axis=-1)
    # (-1,1) → (0,255)
    vis = ((normal_img + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    # RGB → BGR
    bgr = vis[..., ::-1].copy()
    bgr[~valid] = 0
    return bgr


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """在图像左上角添加白色文字标签。"""
    out = img.copy()
    cv2.putText(out, text, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def show_windows(vis_list: list):
    """
    弹出 4 个 OpenCV 窗口显示可视化结果。
    按任意键关闭所有窗口。
    """
    scale_h = 3   # 垂直放大倍数（64行 → 192行，更易查看）

    for title, img in vis_list:
        h, w = img.shape[:2]
        resized = cv2.resize(img, (w, h * scale_h),
                             interpolation=cv2.INTER_NEAREST)
        labeled = add_label(resized, title)
        cv2.imshow(title, labeled)

    print("\n[可视化] 按任意键关闭所有窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────
# 4. 测试点云生成
# ─────────────────────────────────────────────────────────────

def generate_test_cloud(n_ground: int = 30000,
                        n_wall:   int = 10000,
                        n_beam:   int = 3000) -> np.ndarray:
    """
    生成模拟室内场景点云（地面 + 正面墙 + 横梁），
    格式 (N, 4): [x, y, z, intensity]
    """
    rng = np.random.default_rng(42)
    pts = []

    # 地面 (z ≈ -0.35m, 大范围水平面, 低光强)
    gx = rng.uniform(-20, 20, n_ground)
    gy = rng.uniform(-20, 20, n_ground)
    gz = rng.normal(-0.35, 0.02, n_ground)
    gi = rng.uniform(0.1, 0.3, n_ground)
    pts.append(np.column_stack([gx, gy, gz, gi]))

    # 正面墙 (x ≈ 15m, 垂直平面, 中等光强)
    wx = rng.normal(15.0, 0.05, n_wall)
    wy = rng.uniform(-8, 8, n_wall)
    wz = rng.uniform(-0.5, 4.0, n_wall)
    wi = rng.uniform(0.6, 0.9, n_wall)
    pts.append(np.column_stack([wx, wy, wz, wi]))

    # 侧墙 (y ≈ -8m, 垂直平面)
    sx = rng.uniform(2, 14, n_beam)
    sy = rng.normal(-8.0, 0.05, n_beam)
    sz = rng.uniform(-0.5, 4.0, n_beam)
    si = rng.uniform(0.5, 0.8, n_beam)
    pts.append(np.column_stack([sx, sy, sz, si]))

    # 天花板横梁 (z ≈ 3.5m, 水平延伸, 低光强)
    bx = rng.uniform(3, 13, n_beam)
    by = rng.normal(0.0, 0.1, n_beam)
    bz = rng.normal(3.5, 0.04, n_beam)
    bi = rng.uniform(0.2, 0.4, n_beam)
    pts.append(np.column_stack([bx, by, bz, bi]))

    return np.vstack(pts).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 5. 点云文件加载（多格式自动识别）
# ─────────────────────────────────────────────────────────────

# Mid-360 通过 livox_ros_driver2 导出的自定义二进制格式
# 每点结构体（40 字节）：
#   float x, y, z        (12 B)
#   float intensity       (4 B)  — 反射强度 [0, 255]
#   uint32 offset_time    (4 B)  — 相对帧起始时间 (ns)
#   uint8  line           (1 B)  — 扫描线号
#   uint8  tag            (1 B)  — 点标签 (0=正常)
#   uint16 reserved       (2 B)
#   double timestamp      (8 B)  — 绝对时间戳 (s)
#   uint32 scan_id        (4 B)
#   uint32 reserved2      (4 B)
# ─────────────────────────────────────────────────────────────
_MID360_DTYPE = np.dtype([
    ('x',           np.float32),
    ('y',           np.float32),
    ('z',           np.float32),
    ('intensity',   np.float32),
    ('offset_time', np.uint32),
    ('line',        np.uint8),
    ('tag',         np.uint8),
    ('reserved',    np.uint16),
    ('timestamp',   np.float64),
    ('scan_id',     np.uint32),
    ('reserved2',   np.uint32),
])  # 总计 40 字节/点


def _load_pcd(path: str) -> np.ndarray:
    """
    解析 PCL PCD 格式（ascii / binary / binary_compressed）。
    返回 (N, 4) float32: [x, y, z, intensity]
    intensity 字段缺失时填 0。
    """
    with open(path, 'rb') as f:
        raw = f.read()

    # ── 解析文本头部 ─────────────────────────────────────────
    header_lines = []
    header_bytes = 0
    pos = 0
    while pos < len(raw):
        end = raw.index(b'\n', pos)
        line = raw[pos:end].decode('ascii', errors='ignore').strip()
        header_lines.append(line)
        pos = end + 1
        if line.startswith('DATA'):
            header_bytes = pos
            break

    meta = {}
    for line in header_lines:
        if line.startswith('#') or not line:
            continue
        key, *vals = line.split()
        meta[key.upper()] = vals

    fields    = [f.lower() for f in meta.get('FIELDS', ['x', 'y', 'z'])]
    sizes     = [int(s)    for s in meta.get('SIZE',   ['4'] * len(fields))]
    types     = meta.get('TYPE', ['F'] * len(fields))
    n_points  = int(meta.get('POINTS', ['0'])[0])
    data_type = meta.get('DATA', ['binary'])[0].lower()

    # ── 构造 numpy dtype ─────────────────────────────────────
    _type_map = {'F': {4: np.float32, 8: np.float64},
                 'I': {1: np.int8,  2: np.int16,  4: np.int32,  8: np.int64},
                 'U': {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}}
    dt_fields = []
    for fname, ftype, fsize in zip(fields, types, sizes):
        np_type = _type_map.get(ftype, {}).get(int(fsize), np.float32)
        dt_fields.append((fname, np_type))
    dtype = np.dtype(dt_fields)

    # ── 读取点数据 ───────────────────────────────────────────
    if data_type == 'binary':
        body = raw[header_bytes:]
        needed = n_points * dtype.itemsize
        data = np.frombuffer(body[:needed], dtype=dtype)
    elif data_type == 'ascii':
        import io
        body = raw[header_bytes:].decode('ascii', errors='ignore')
        data = np.loadtxt(io.StringIO(body), max_rows=n_points)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # ascii 模式直接用列索引
        x_col = fields.index('x') if 'x' in fields else 0
        y_col = fields.index('y') if 'y' in fields else 1
        z_col = fields.index('z') if 'z' in fields else 2
        i_col = fields.index('intensity') if 'intensity' in fields else -1
        x = data[:, x_col].astype(np.float32)
        y = data[:, y_col].astype(np.float32)
        z = data[:, z_col].astype(np.float32)
        intens = data[:, i_col].astype(np.float32) if i_col >= 0 \
                 else np.zeros(len(x), np.float32)
        if intens.max() > 1.5:
            intens = intens / 255.0
        print(f"[加载] PCD ascii  {len(x)} 点  字段: {fields}")
        return np.column_stack([x, y, z, intens]).astype(np.float32)
    elif data_type == 'binary_compressed':
        import struct, zlib
        body = raw[header_bytes:]
        comp_size, decomp_size = struct.unpack('<II', body[:8])
        decompressed = zlib.decompress(body[8:8 + comp_size], -15)
        # binary_compressed 以列优先存储（每字段连续）
        pts_out = np.zeros((n_points, 4), dtype=np.float32)
        offset = 0
        col_idx = 0
        for fname, ftype, fsize in zip(fields, types, sizes):
            np_type = _type_map.get(ftype, {}).get(int(fsize), np.float32)
            col = np.frombuffer(
                decompressed[offset:offset + n_points * int(fsize)],
                dtype=np_type).astype(np.float32)
            if fname == 'x':     pts_out[:, 0] = col
            elif fname == 'y':   pts_out[:, 1] = col
            elif fname == 'z':   pts_out[:, 2] = col
            elif fname == 'intensity':
                pts_out[:, 3] = col / 255.0 if col.max() > 1.5 else col
            offset += n_points * int(fsize)
            col_idx += 1
        print(f"[加载] PCD binary_compressed  {n_points} 点  字段: {fields}")
        return pts_out
    else:
        raise ValueError(f"不支持的 PCD DATA 类型: {data_type}")

    # ── binary 模式提取 x,y,z,intensity ─────────────────────
    x = data['x'].astype(np.float32) if 'x' in fields else np.zeros(n_points, np.float32)
    y = data['y'].astype(np.float32) if 'y' in fields else np.zeros(n_points, np.float32)
    z = data['z'].astype(np.float32) if 'z' in fields else np.zeros(n_points, np.float32)

    if 'intensity' in fields:
        intens = data['intensity'].astype(np.float32)
        if intens.max() > 1.5:
            intens = intens / 255.0
    else:
        intens = np.zeros(n_points, np.float32)

    print(f"[加载] PCD binary  {n_points} 点  字段: {fields}")
    return np.column_stack([x, y, z, intens]).astype(np.float32)


def load_bin(path: str) -> np.ndarray:
    """
    自动识别点云文件格式，返回 (N, 4) float32：[x, y, z, intensity]

    支持格式：
      - PCD (ascii / binary / binary_compressed)  — PointCloudXYZI 等
      - Livox Mid-360 自定义二进制               — 每点 40B 结构体
      - KITTI / Velodyne .bin                    — 每点 4×float32
      - 速腾 RS .bin                             — 每点 5×float32
      - 6 字段格式                               — 每点 6×float32
    """
    import os
    # ── PCD 格式：检测文件头魔数 ─────────────────────────────
    with open(path, 'rb') as f:
        magic = f.read(11)
    if magic.startswith(b'# .PCD') or magic.startswith(b'VERSION'):
        return _load_pcd(path)

    file_size = os.path.getsize(path)
    raw = np.fromfile(path, dtype=np.uint8)

    # ── 优先尝试 Mid-360 40B 结构体 ─────────────────────────
    stride = _MID360_DTYPE.itemsize   # 40
    if file_size % stride == 0 and file_size // stride > 10:
        data = raw.view(_MID360_DTYPE)
        # 合理性检验：Mid-360 正常点 tag==0，且 xyz 在有效室内范围内
        #   随机 float 解析出的 tag 字段会是随机 uint8，极少全为 0
        tag_zero_ratio = (data['tag'] == 0).mean()
        xyz_ok = (np.abs(data['x']) < 200).mean() > 0.9
        if tag_zero_ratio > 0.8 and xyz_ok:
            valid = data['tag'] == 0
            pts = np.column_stack([
                data['x'].astype(np.float32),
                data['y'].astype(np.float32),
                data['z'].astype(np.float32),
                (data['intensity'] / 255.0).astype(np.float32),
            ])[valid]
            print(f"[加载] Mid-360 自定义格式  {len(data)} 点 → 有效 {len(pts)} 点")
            return pts

    # ── 回退：纯 float32，按字段数自动判断 ──────────────────
    n_floats = file_size // 4
    remainder = file_size % 4

    data_f = np.frombuffer(raw[:file_size - remainder], dtype=np.float32)

    for fields in (4, 5, 6, 3):
        if n_floats % fields == 0:
            arr = data_f.reshape(-1, fields)
            # 简单合理性检验：xyz 范围在 [-200, 200] 内
            xyz = arr[:, :3]
            if np.all(np.abs(xyz) < 500):
                intensity = arr[:, 3] if fields >= 4 else np.zeros(len(arr), np.float32)
                # intensity 归一化：若 >1 则假定是 [0,255] 原始值
                if intensity.max() > 1.5:
                    intensity = intensity / 255.0
                pts = np.column_stack([xyz, intensity]).astype(np.float32)
                fmt_name = {4: "KITTI/Velodyne (4字段)",
                            5: "速腾 RS (5字段)",
                            6: "6字段",
                            3: "3字段(无intensity)"}[fields]
                print(f"[加载] {fmt_name}  共 {len(pts)} 点")
                return pts

    raise ValueError(
        f"无法识别格式: size={file_size}B, n_floats={n_floats}\n"
        f"  可整除字段数: " +
        ", ".join(str(f) for f in (3,4,5,6,7,8) if n_floats % f == 0) +
        "\n  请用 --fields N 手动指定"
    )


# ─────────────────────────────────────────────────────────────
# 6. 主流程
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="点云球面投影 + 法向量计算可视化")
    parser.add_argument("--input",     type=str,   default="",
                        help="点云文件路径（KITTI .bin / 速腾 .bin / Mid-360 自定义二进制）")
    parser.add_argument("--fields",    type=int,   default=0,
                        help="强制指定每点字段数（0=自动识别）")
    parser.add_argument("--height",    type=int,   default=H)
    parser.add_argument("--width",     type=int,   default=W)
    parser.add_argument("--fov-up",    type=float, default=FOV_UP_DEG)
    parser.add_argument("--fov-down",  type=float, default=FOV_DOWN_DEG)
    parser.add_argument("--min-range", type=float, default=MIN_RANGE)
    parser.add_argument("--max-range", type=float, default=MAX_RANGE)
    parser.add_argument("--save",      type=str,   default="",
                        help="保存拼接图像的路径（如 output.png），不填则只弹窗")
    args = parser.parse_args()

    # ── 加载 / 生成点云 ──────────────────────────────────────
    if args.input:
        if args.fields > 0:
            # 手动指定字段数
            raw = np.fromfile(args.input, dtype=np.float32)
            arr = raw[:len(raw) // args.fields * args.fields].reshape(-1, args.fields)
            intensity = arr[:, 3] if args.fields >= 4 else np.zeros(len(arr), np.float32)
            if intensity.max() > 1.5:
                intensity = intensity / 255.0
            pts = np.column_stack([arr[:, :3], intensity]).astype(np.float32)
            print(f"[加载] 手动 {args.fields} 字段  共 {len(pts)} 点")
        else:
            pts = load_bin(args.input)
        print(f"       xyz range: x=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}]"
              f" y=[{pts[:,1].min():.2f},{pts[:,1].max():.2f}]"
              f" z=[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]"
              f" intensity=[{pts[:,3].min():.3f},{pts[:,3].max():.3f}]")
    else:
        pts = generate_test_cloud()
        print(f"[测试点云] 生成 {len(pts)} 点（地面+正面墙+侧墙+横梁）")

    # ── 球面投影 ─────────────────────────────────────────────
    range_img, intensity_img, xyz_img, pixel_idx = spherical_projection(
        pts,
        h=args.height, w=args.width,
        fov_up_deg=args.fov_up, fov_down_deg=args.fov_down,
        min_range=args.min_range, max_range=args.max_range,
    )

    valid_pixels = (range_img > 0).sum()
    total_pixels = args.height * args.width
    print(f"[投影] 有效像素: {valid_pixels}/{total_pixels} "
          f"({100*valid_pixels/total_pixels:.1f}%)  "
          f"range=[{range_img[range_img>0].min():.2f}, "
          f"{range_img[range_img>0].max():.2f}] m")

    # ── 法向量计算 ───────────────────────────────────────────
    normal_img = compute_normals(xyz_img, range_img)

    valid_normals = np.any(normal_img != 0, axis=-1).sum()
    print(f"[法向量] 有效像素: {valid_normals}/{total_pixels} "
          f"({100*valid_normals/total_pixels:.1f}%)")

    # ── 生成可视化图像 ───────────────────────────────────────
    vis_depth    = make_depth_vis(range_img)
    vis_intens   = make_intensity_vis(intensity_img)
    vis_xyz      = make_xyz_vis(xyz_img, range_img)
    vis_normals  = make_normal_vis(normal_img)

    vis_list = [
        ("Stream 1: Depth (Range)",          vis_depth),
        ("Stream 1: Intensity",              vis_intens),
        ("Stream 2: Geometry (XYZ)",         vis_xyz),
        ("Stream 2: Direction Vectors (Normals)", vis_normals),
    ]

    # ── 打印各窗口含义 ───────────────────────────────────────
    print("""
┌─────────────────────────────────────────────────────────┐
│  窗口说明                                               │
├────────────────────────────┬────────────────────────────┤
│ Stream 1: Depth (Range)    │ JET 伪彩色深度图            │
│                            │ 蓝=近  红=远               │
├────────────────────────────┼────────────────────────────┤
│ Stream 1: Intensity        │ 灰度光强图                  │
│                            │ 亮=强反射  暗=弱反射        │
├────────────────────────────┼────────────────────────────┤
│ Stream 2: Geometry (XYZ)   │ RGB=(X,Y,Z) 坐标编码        │
│                            │ 颜色变化反映空间位置        │
├────────────────────────────┼────────────────────────────┤
│ Stream 2: Direction Vectors│ RGB=(nx,ny,nz) 法向量       │
│                            │ 水平面→紫  垂直面→绿/蓝    │
└────────────────────────────┴────────────────────────────┘
""")

    # ── 保存拼接图 ───────────────────────────────────────────
    if args.save:
        scale_h = 3
        rows = []
        for _, img in vis_list:
            h_img, w_img = img.shape[:2]
            rows.append(cv2.resize(img, (w_img, h_img * scale_h),
                                   interpolation=cv2.INTER_NEAREST))
        combined = np.vstack(rows)
        cv2.imwrite(args.save, combined)
        print(f"[保存] {args.save}")

    # ── 弹窗显示 ─────────────────────────────────────────────
    show_windows(vis_list)


if __name__ == "__main__":
    main()
