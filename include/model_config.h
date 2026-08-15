#pragma once
// ============================================================
// model_config.h — Dual-Stream RangeNet Lite v2 全局配置
// ============================================================
#include <cstdint>
#include <string>

namespace rangenet {

// ────────────────────────────────────────────────────────────
// 语义类别 (9类)
// ────────────────────────────────────────────────────────────
constexpr int NUM_CLASSES = 9;

static const char* CLASS_NAMES[NUM_CLASSES] = {
    "background",    // 0
    "ground",        // 1  地面    — nz≈+1, z低
    "roof",          // 2  屋顶    — nz≈+1, z高
    "side_facade",   // 3  侧立面  — nz≈0, nx主导
    "front_facade",  // 4  前立面  — nz≈0, ny主导
    "beam",          // 5  横梁    — 水平延伸
    "column",        // 6  柱子    — 垂直延伸
    "window",        // 7  窗户    — 低intensity
    "dynamic",       // 8  动态目标
};

// RGB 颜色 (用于 PLY 可视化)
static const uint8_t CLASS_COLORS[NUM_CLASSES][3] = {
    {  0,   0,   0},   // background
    {128,  64, 128},   // ground
    { 70,  70,  70},   // roof
    {190, 153, 153},   // side_facade
    {153, 153, 190},   // front_facade
    {250, 170,  30},   // beam
    {220, 220,   0},   // column
    {107, 142,  35},   // window
    {220,  20,  60},   // dynamic
};

// ────────────────────────────────────────────────────────────
// 模型输入
// ────────────────────────────────────────────────────────────
constexpr int INPUT_HEIGHT   = 64;
constexpr int INPUT_WIDTH    = 1024;

// 8通道输入:
//   Stream-1 (Range Stream)   : ch[0]=range,  ch[1]=intensity
//   Stream-2 (Geometry Stream): ch[2]=nx, ch[3]=ny, ch[4]=nz,
//                               ch[5]=x,  ch[6]=y,  ch[7]=z
constexpr int INPUT_CHANNELS = 8;
constexpr int STREAM1_CH     = 2;   // range, intensity
constexpr int STREAM2_CH     = 6;   // nx, ny, nz, x, y, z

// ────────────────────────────────────────────────────────────
// 传感器 FOV 参数 — Livox Mid-360 (默认)
//   垂直 FOV: -7° ~ +52° (仰角向上为正)
//   水平 FOV: 360° (全向扫描)
//   投影: 仅保留 FOV_DOWN_DEG ~ FOV_UP_DEG 范围
// ────────────────────────────────────────────────────────────
constexpr float FOV_UP_DEG   = 52.0f;
constexpr float FOV_DOWN_DEG = -7.0f;
constexpr float MIN_RANGE    =  0.1f;   // Mid-360 最小测距 0.1m
constexpr float MAX_RANGE    = 40.0f;   // Mid-360 典型有效测距 40m

// ────────────────────────────────────────────────────────────
// 传感器预设 — 可通过 --sensor 参数在运行时选择
// ────────────────────────────────────────────────────────────
struct SensorPreset {
    const char* name;
    float fov_up_deg;
    float fov_down_deg;
    float max_range;
    int   n_keyframe_accum;  // 1 = 单帧直接推理（机械式雷达）
};

// 所有预设使用相同 MIN_RANGE = 0.1f
static const SensorPreset SENSOR_PRESETS[] = {
    // name                fov_up  fov_down  max_range  accum
    { "mid360",            52.0f,  -7.0f,    40.0f,     20  },  // Livox Mid-360（默认）
    { "rs-helios",         15.0f, -15.0f,    30.0f,      1  },  // 速腾 RS-Helios 32线
    { "rs-helios16p",      15.0f, -15.0f,    20.0f,      1  },  // 速腾 RS-Helios-16P
    { "rs-ruby",           15.0f, -25.0f,    30.0f,      1  },  // 速腾 RS-Ruby 128线
    { "rs-lidar16",        15.0f, -15.0f,    20.0f,      1  },  // 速腾 RS-LiDAR-16
    { "vlp16",             15.0f, -15.0f,    20.0f,      1  },  // Velodyne VLP-16
    { "hdl32",             10.67f,-30.67f,   30.0f,      1  },  // Velodyne HDL-32E
    { "hdl64",              2.0f, -24.8f,    50.0f,      1  },  // Velodyne HDL-64E
    { "vls128",            15.0f, -25.0f,    50.0f,      1  },  // Velodyne VLS-128
};
static constexpr int NUM_SENSOR_PRESETS =
    static_cast<int>(sizeof(SENSOR_PRESETS) / sizeof(SENSOR_PRESETS[0]));

/** 按名称查找预设，未找到返回 nullptr */
inline const SensorPreset* findSensorPreset(const std::string& name) {
    for (int i = 0; i < NUM_SENSOR_PRESETS; ++i)
        if (name == SENSOR_PRESETS[i].name) return &SENSOR_PRESETS[i];
    return nullptr;
}

// ────────────────────────────────────────────────────────────
// 特征归一化参数 (z-score, 与 Python export 一致)
// 通道顺序: range, intensity, nx, ny, nz, x, y, z
// 统计来源: UBPC-9 训练集, 传感器为 Livox Mid-360 (安装高度≈0.35m)
// ────────────────────────────────────────────────────────────
static const float CHANNEL_MEAN[INPUT_CHANNELS] = {
     8.50f,   // range      (m)   Mid-360场景典型测距8.5m
     0.45f,   // intensity         Mid-360返回强度归一化均值
     0.00f,   // normal_x   单位法向量均值≈0
     0.00f,   // normal_y
     0.00f,   // normal_z
     0.00f,   // x
     0.00f,   // y
    -0.35f,   // z          传感器离地约0.35m (NUC机器人平台)
};

static const float CHANNEL_STD[INPUT_CHANNELS] = {
     7.20f,   // range
     0.22f,   // intensity
     0.50f,   // normal_x  (单位球面, 标准差≈0.5)
     0.50f,   // normal_y
     0.50f,   // normal_z
     8.60f,   // x          城市建筑场景空间范围约±8.6m
     8.60f,   // y
     1.80f,   // z          建筑高度变化范围约±1.8m
};

// ────────────────────────────────────────────────────────────
// 关键帧构建参数 — 多帧叠加
//   Mid-360 非重复性扫描单帧点数约 ~1000~3000 点 (10ms积分)
//   叠加 N_KEYFRAME_ACCUM 帧后得到约 20~60k 点的稠密关键帧
//   机械式雷达（速腾/Velodyne）单帧已足够，设为 1 可直接推理
// ────────────────────────────────────────────────────────────
constexpr int   N_KEYFRAME_ACCUM     = 20;    // 叠加帧数 (对应约200ms时间窗口)
constexpr float KEYFRAME_DIST_M      = 0.3f;  // 关键帧触发: 行进距离阈值 (m)
constexpr float KEYFRAME_ANGLE_DEG   = 5.0f;  // 关键帧触发: 偏航角变化阈值 (°)
constexpr float MOTION_DIST_CELL     = 0.05f; // 运动补偿: 体素格大小 (m), 去除移动点
static const char* MODEL_PATH = "models/dual_rangenet_lite.onnx";

constexpr int  DEFAULT_INTRA_OP_THREADS = 4;
constexpr int  WARMUP_RUNS              = 5;
constexpr int  KNN_K_NEIGHBORS          = 5;

} // namespace rangenet
