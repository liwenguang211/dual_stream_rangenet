// ============================================================
// RangeNetInferencer.cpp — Dual-Stream RangeNet Lite v2 推理实现
//
// v2 架构变化:
//  INPUT_CHANNELS = 8 (range, intensity, nx, ny, nz, x, y, z)
//  Stream-1 ch[0:2] = [range, intensity]  (外观/材质)
//  Stream-2 ch[2:8] = [nx,ny,nz, x,y,z]  (几何法向量+坐标)
//
// 关键实现说明:
//  - CHW 布局: img[c*H*W + pixel]，与 PyTorch ONNX 导出一致
//  - project() 内置法向量计算（深度图梯度叉积）
//  - 8通道 z-score 归一化（CHANNEL_MEAN / CHANNEL_STD）
//  - 输出解析 logits[c*HW + i]，CHW 格式
//  - KNN 填充使用体素哈希 O(N) 平均复杂度
// ============================================================
#include "RangeNetInferencer.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include <unordered_map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace rangenet {

// ────────────────────────────────────────────────────────────
// 计时工具
// ────────────────────────────────────────────────────────────
static inline double nowMs() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
               clk::now().time_since_epoch())
        .count();
}

// ────────────────────────────────────────────────────────────
// RangeImageProcessor
// ────────────────────────────────────────────────────────────

RangeImageProcessor::RangeImageProcessor(int H, int W,
                                         float fov_up_deg,
                                         float fov_down_deg)
    : H_(H), W_(W),
      fov_up_  (fov_up_deg   * static_cast<float>(M_PI) / 180.f),
      fov_down_(fov_down_deg * static_cast<float>(M_PI) / 180.f),
      fov_     (std::fabs(fov_up_) + std::fabs(fov_down_)),
      max_range_(MAX_RANGE)   // 默认值，可由 setMaxRange() 覆盖
{}

void RangeImageProcessor::project(const std::vector<float>& pts,
                                  std::vector<float>&       img,
                                  std::vector<int32_t>&     idx) const
{
    // ── 8通道 CHW 范围图像 ────────────────────────────────────
    // ch[0]=range  ch[1]=intensity
    // ch[2]=nx     ch[3]=ny     ch[4]=nz   (法向量, Pass-2计算)
    // ch[5]=x      ch[6]=y      ch[7]=z

    const int HW = H_ * W_;
    img.assign(INPUT_CHANNELS * HW, 0.f);
    idx.assign(HW, -1);

    std::vector<float> depth_map(HW, std::numeric_limits<float>::max());

    // ── Pass-1: 球面投影, 填充 range/intensity/xyz ────────────
    const size_t n = pts.size() / 4;
    for (size_t i = 0; i < n; ++i) {
        const float x = pts[i * 4 + 0];
        const float y = pts[i * 4 + 1];
        const float z = pts[i * 4 + 2];
        const float s = pts[i * 4 + 3];  // intensity

        const float r = std::sqrt(x*x + y*y + z*z);
        if (r < MIN_RANGE || r > max_range_) continue;

        const float yaw   = std::atan2(y, x);
        const float pitch = std::asin(z / r);

        const float vf = (1.f - (pitch - fov_down_) / fov_) * H_;
        const float uf = (0.5f * (1.f - yaw / static_cast<float>(M_PI))) * W_;

        const int vi = std::max(0, std::min(H_ - 1, static_cast<int>(vf)));
        const int ui = std::max(0, std::min(W_ - 1, static_cast<int>(uf)));
        const int pi = vi * W_ + ui;

        if (r >= depth_map[pi]) continue;
        depth_map[pi] = r;
        idx[pi]       = static_cast<int32_t>(i);

        // ★ CHW: img[channel * HW + pixel]
        img[0 * HW + pi] = r;    // range
        img[1 * HW + pi] = s;    // intensity
        // nx,ny,nz (ch2-4) 在 Pass-2 计算
        img[5 * HW + pi] = x;    // x
        img[6 * HW + pi] = y;    // y
        img[7 * HW + pi] = z;    // z
    }

    // ── Pass-2: 从深度图计算法向量 (有限差分叉积) ────────────
    // 对每个有效像素 (r,c):
    //   dP_dr = P(r+1,c) - P(r-1,c)   (垂直梯度)
    //   dP_dc = P(r,c+1) - P(r,c-1)   (水平梯度)
    //   N = normalize(dP_dr × dP_dc)
    float* nx_ch = img.data() + 2 * HW;
    float* ny_ch = img.data() + 3 * HW;
    float* nz_ch = img.data() + 4 * HW;
    const float* px_ch = img.data() + 5 * HW;
    const float* py_ch = img.data() + 6 * HW;
    const float* pz_ch = img.data() + 7 * HW;

    for (int r = 1; r < H_ - 1; ++r) {
        for (int c = 1; c < W_ - 1; ++c) {
            const int pi   = r * W_ + c;
            if (depth_map[pi] >= std::numeric_limits<float>::max()) continue;

            // 垂直邻居 (上下行)
            const int pi_u = (r - 1) * W_ + c;
            const int pi_d = (r + 1) * W_ + c;
            // 水平邻居 (左右列)
            const int pi_l = r * W_ + (c - 1);
            const int pi_r = r * W_ + (c + 1);

            // 梯度向量 (跳过无效邻居)
            bool valid_ud = (depth_map[pi_u] < 1e9f && depth_map[pi_d] < 1e9f);
            bool valid_lr = (depth_map[pi_l] < 1e9f && depth_map[pi_r] < 1e9f);
            if (!valid_ud || !valid_lr) continue;

            float dxr = px_ch[pi_d] - px_ch[pi_u];
            float dyr = py_ch[pi_d] - py_ch[pi_u];
            float dzr = pz_ch[pi_d] - pz_ch[pi_u];

            float dxc = px_ch[pi_r] - px_ch[pi_l];
            float dyc = py_ch[pi_r] - py_ch[pi_l];
            float dzc = pz_ch[pi_r] - pz_ch[pi_l];

            // 叉积 N = dP_dr × dP_dc
            float nx = dyr * dzc - dzr * dyc;
            float ny = dzr * dxc - dxr * dzc;
            float nz = dxr * dyc - dyr * dxc;
            float len = std::sqrt(nx*nx + ny*ny + nz*nz);

            if (len < 1e-6f) continue;
            nx_ch[pi] = nx / len;
            ny_ch[pi] = ny / len;
            nz_ch[pi] = nz / len;
        }
    }

    // ── z-score 归一化 (仅有效像素, 8通道) ───────────────────
    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        const float mu  = CHANNEL_MEAN[c];
        const float sig = CHANNEL_STD[c];
        float* ch = img.data() + c * HW;
        for (int i = 0; i < HW; ++i) {
            if (depth_map[i] < std::numeric_limits<float>::max())
                ch[i] = (ch[i] - mu) / sig;
        }
    }
}

void RangeImageProcessor::backProject(
    const std::vector<int32_t>& pix_labels,
    const std::vector<float>&   pix_conf,
    const std::vector<int32_t>& idx,
    size_t                      n_pts,
    std::vector<int32_t>&       labels,
    std::vector<float>&         confs) const
{
    labels.assign(n_pts, 0);
    confs .assign(n_pts, 0.f);

    const int HW = H_ * W_;
    for (int i = 0; i < HW; ++i) {
        const int32_t pt = idx[i];
        if (pt < 0 || static_cast<size_t>(pt) >= n_pts) continue;
        labels[pt] = pix_labels[i];
        confs [pt] = pix_conf  [i];
    }
}

void RangeImageProcessor::knnFill(const std::vector<float>& pts,
                                  std::vector<int32_t>&     labels,
                                  std::vector<float>&       confs,
                                  int                       K) const
{
    // ── 体素哈希网格 KNN 填充 (O(N) 平均复杂度) ─────────────
    // 替换原始 O(N×M) 暴力搜索，将后处理从 ~2s 降到 <5ms
    const size_t n = pts.size() / 4;

    // 收集已标记点
    std::vector<size_t> known;
    known.reserve(n);
    for (size_t i = 0; i < n; ++i)
        if (confs[i] > 0.f) known.push_back(i);

    if (known.empty()) return;

    // 体素格大小 (米): 足够大以保证每格有若干邻居
    constexpr float CELL = 1.0f;
    constexpr float INV_CELL = 1.0f / CELL;

    // 体素哈希键 (int32 x/y/z → uint64)
    auto voxelKey = [](int ix, int iy, int iz) -> uint64_t {
        // 偏移 1000 保证非负，裁剪在 20bit 内
        uint64_t ux = static_cast<uint64_t>((ix + 2000) & 0xFFFFF);
        uint64_t uy = static_cast<uint64_t>((iy + 2000) & 0xFFFFF);
        uint64_t uz = static_cast<uint64_t>((iz + 2000) & 0xFFFFF);
        return ux | (uy << 20) | (uz << 40);
    };

    // 建立哈希表: voxel_key → {point_index, ...}
    std::unordered_map<uint64_t, std::vector<size_t>> grid;
    grid.reserve(known.size() * 2);
    for (size_t j : known) {
        int ix = static_cast<int>(pts[j*4+0] * INV_CELL);
        int iy = static_cast<int>(pts[j*4+1] * INV_CELL);
        int iz = static_cast<int>(pts[j*4+2] * INV_CELL);
        grid[voxelKey(ix, iy, iz)].push_back(j);
    }

    // 对每个未标记点搜索 3×3×3=27 邻格
    for (size_t i = 0; i < n; ++i) {
        if (confs[i] > 0.f) continue;

        const float xi = pts[i*4+0];
        const float yi = pts[i*4+1];
        const float zi = pts[i*4+2];
        const int   ox = static_cast<int>(xi * INV_CELL);
        const int   oy = static_cast<int>(yi * INV_CELL);
        const int   oz = static_cast<int>(zi * INV_CELL);

        float best_d2[NUM_CLASSES];
        int   votes  [NUM_CLASSES] = {};
        std::fill(best_d2, best_d2 + NUM_CLASSES,
                  std::numeric_limits<float>::max());

        // 搜索半径最多扩展 3 次直到找到邻居
        for (int radius = 1; radius <= 3; ++radius) {
            bool found = false;
            for (int dx = -radius; dx <= radius; ++dx)
            for (int dy = -radius; dy <= radius; ++dy)
            for (int dz = -radius; dz <= radius; ++dz) {
                auto it = grid.find(voxelKey(ox+dx, oy+dy, oz+dz));
                if (it == grid.end()) continue;
                for (size_t j : it->second) {
                    float ddx = xi - pts[j*4+0];
                    float ddy = yi - pts[j*4+1];
                    float ddz = zi - pts[j*4+2];
                    float d2  = ddx*ddx + ddy*ddy + ddz*ddz;
                    int   lb  = labels[j];
                    if (d2 < best_d2[lb]) {
                        best_d2[lb] = d2;
                        votes  [lb]++;
                        found = true;
                    }
                }
            }
            if (found) break;
        }

        labels[i] = static_cast<int32_t>(
            std::max_element(votes, votes + NUM_CLASSES) - votes);
        confs [i] = 0.5f;
    }
}

// ────────────────────────────────────────────────────────────
// RangeNetInferencer
// ────────────────────────────────────────────────────────────

RangeNetInferencer::RangeNetInferencer(const std::string& model_path,
                                       bool  use_gpu, int device_id,
                                       bool  use_trt, bool fp16,
                                       int   n_threads,
                                       float fov_up_deg,
                                       float fov_down_deg,
                                       float max_range,
                                       int   n_keyframe_accum)
    : env_(ORT_LOGGING_LEVEL_WARNING, "DualRangeNet"),
      model_path_      (model_path),
      use_gpu_         (use_gpu),
      device_id_       (device_id),
      use_trt_         (use_trt),
      fp16_            (fp16),
      n_threads_       (n_threads),
      fov_up_deg_      (fov_up_deg),
      fov_down_deg_    (fov_down_deg),
      max_range_       (max_range),
      n_keyframe_accum_(n_keyframe_accum)
{
    proc_ = std::make_unique<RangeImageProcessor>(
        INPUT_HEIGHT, INPUT_WIDTH, fov_up_deg_, fov_down_deg_);

    // 将运行时 max_range 写入 processor（project() 使用成员变量读取）
    proc_->setMaxRange(max_range_);

    const int HW = INPUT_HEIGHT * INPUT_WIDTH;
    img_buf_.resize(INPUT_CHANNELS * HW, 0.f);
    idx_buf_.resize(HW, -1);

    std::cout << "[RangeNet] 传感器配置: FOV " << fov_up_deg_ << "°/" << fov_down_deg_
              << "°  max_range=" << max_range_ << "m"
              << "  keyframe_accum=" << n_keyframe_accum_ << std::endl;
}

void RangeNetInferencer::buildSession() {
    opts_.SetIntraOpNumThreads(n_threads_);
    opts_.SetInterOpNumThreads(1);
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts_.EnableMemPattern();
    opts_.EnableCpuMemArena();

    // 缓存优化后图
    std::string cached = model_path_ + ".opt.ort";
    opts_.SetOptimizedModelFilePath(cached.c_str());

    // ── 执行提供者 ────────────────────────────────────────────
    if (use_trt_) {
        OrtTensorRTProviderOptions trt{};
        trt.device_id              = device_id_;
        trt.trt_max_workspace_size = 1ULL << 30;
        trt.trt_fp16_enable        = fp16_ ? 1 : 0;
        trt.trt_engine_cache_enable = 1;
        trt.trt_engine_cache_path  = "./trt_cache";
        trt.trt_max_partition_iterations = 1000;
        trt.trt_min_subgraph_size  = 5;
        opts_.AppendExecutionProvider_TensorRT(trt);
        std::cout << "[RangeNet] TensorRT provider"
                  << (fp16_ ? " FP16" : "") << std::endl;
    }
    if (use_gpu_) {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id                 = device_id_;
        cuda.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchExhaustive;
        cuda.arena_extend_strategy     = 0;
        cuda.do_copy_in_default_stream = 1;
        opts_.AppendExecutionProvider_CUDA(cuda);
        std::cout << "[RangeNet] CUDA provider (device=" << device_id_ << ")" << std::endl;
    }

    session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), opts_);

    // ★ 使用非废弃 API: GetInputNameAllocated / GetOutputNameAllocated
    Ort::AllocatorWithDefaultOptions alloc;

    in_names_.clear();
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
        auto ptr = session_->GetInputNameAllocated(i, alloc);
        in_names_.emplace_back(ptr.get());
    }
    out_names_.clear();
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        auto ptr = session_->GetOutputNameAllocated(i, alloc);
        out_names_.emplace_back(ptr.get());
    }

    // 缓存稳定的 const char* 指针
    in_cstr_.clear();
    for (auto& s : in_names_)  in_cstr_.push_back(s.c_str());
    out_cstr_.clear();
    for (auto& s : out_names_) out_cstr_.push_back(s.c_str());
}

void RangeNetInferencer::warmup() {
    std::cout << "[RangeNet] 预热 " << WARMUP_RUNS << " 次..." << std::flush;
    const int HW = INPUT_HEIGHT * INPUT_WIDTH;
    std::vector<float> dummy(INPUT_CHANNELS * HW, 0.f);
    for (int i = 0; i < WARMUP_RUNS; ++i) runORT(dummy);
    std::cout << " OK" << std::endl;
}

void RangeNetInferencer::initialize() {
    try {
        buildSession();
        printModelInfo();
        warmup();
        std::cout << "[RangeNet] 初始化完成\n" << std::endl;
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("[ORT] ") + e.what());
    }
}

void RangeNetInferencer::buildRangeImage(const std::vector<float>& pts,
                                          std::vector<float>&       img,
                                          std::vector<int32_t>&     idx) const
{
    proc_->project(pts, img, idx);
}

std::pair<std::vector<int32_t>, std::vector<float>>
RangeNetInferencer::runORT(const std::vector<float>& img)
{
    const int HW = INPUT_HEIGHT * INPUT_WIDTH;

    // ── 构造输入张量 [1, 5, H, W] ─────────────────────────────
    std::array<int64_t, 4> shape{1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_t = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(img.data()), img.size(),
        shape.data(), shape.size());

    // ── 运行推理 ──────────────────────────────────────────────
    auto outs = session_->Run(run_opts_,
        in_cstr_.data(), &in_t, 1,
        out_cstr_.data(), out_cstr_.size());

    // ── 解析 [1, NUM_CLASSES, H, W] CHW ─────────────────────
    // ★ 正确: logits[class_c * HW + pixel_i]
    const float* logits = outs[0].GetTensorData<float>();

    std::vector<int32_t> pix_labels(HW);
    std::vector<float>   pix_conf  (HW);

    for (int i = 0; i < HW; ++i) {
        float max_logit = -1e30f;
        int   max_cls   = 0;
        float vals[NUM_CLASSES];

        for (int c = 0; c < NUM_CLASSES; ++c) {
            vals[c] = logits[c * HW + i];   // ★ CHW 访问
            if (vals[c] > max_logit) { max_logit = vals[c]; max_cls = c; }
        }
        // 数值稳定 softmax 置信度
        float sum_exp = 0.f;
        for (int c = 0; c < NUM_CLASSES; ++c)
            sum_exp += std::exp(vals[c] - max_logit);

        pix_labels[i] = static_cast<int32_t>(max_cls);
        pix_conf  [i] = 1.f / sum_exp;
    }

    return {pix_labels, pix_conf};
}

InferResult RangeNetInferencer::infer(const std::vector<float>& pts) {
    if (pts.size() % 4 != 0)
        throw std::invalid_argument("pts 长度必须是 4 的倍数 [x,y,z,i]");

    const size_t n_pts = pts.size() / 4;
    InferResult res;
    res.stats.num_points = static_cast<int>(n_pts);

    double t0 = nowMs();

    // 1. 预处理
    buildRangeImage(pts, img_buf_, idx_buf_);
    double t1 = nowMs();

    // 2. 推理
    auto [pix_labels, pix_conf] = runORT(img_buf_);
    double t2 = nowMs();

    // 3. 回投 + KNN 填充
    proc_->backProject(pix_labels, pix_conf, idx_buf_, n_pts,
                       res.labels, res.confidences);
    proc_->knnFill(pts, res.labels, res.confidences);
    double t3 = nowMs();

    res.stats.preprocess_ms  = t1 - t0;
    res.stats.inference_ms   = t2 - t1;
    res.stats.postprocess_ms = t3 - t2;
    res.stats.total_ms       = t3 - t0;

    if (!res.confidences.empty()) {
        double s = 0;
        for (float c : res.confidences) s += c;
        res.stats.confidence_mean = s / res.confidences.size();
    }

    {
        std::lock_guard<std::mutex> lk(stat_mu_);
        accum_.preprocess_ms  += res.stats.preprocess_ms;
        accum_.inference_ms   += res.stats.inference_ms;
        accum_.postprocess_ms += res.stats.postprocess_ms;
        accum_.total_ms       += res.stats.total_ms;
        ++total_frames_;
    }
    return res;
}

std::vector<InferResult>
RangeNetInferencer::inferBatch(const std::vector<std::vector<float>>& batch) {
    std::vector<InferResult> results;
    results.reserve(batch.size());
    for (auto& pts : batch) results.push_back(infer(pts));
    return results;
}

void RangeNetInferencer::printModelInfo() const {
    if (!session_) return;
    std::cout << "\n[RangeNet] ─── 模型信息 ───" << std::endl;
    std::cout << "  路径:  " << model_path_ << std::endl;
    std::cout << "  输入(" << in_names_.size() << "):" << std::endl;
    for (size_t i = 0; i < in_names_.size(); ++i) {
        auto sh = session_->GetInputTypeInfo(i)
                           .GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "    " << in_names_[i] << ": [";
        for (size_t j = 0; j < sh.size(); ++j)
            std::cout << sh[j] << (j+1<sh.size()?",":"");
        std::cout << "]" << std::endl;
    }
    std::cout << "  输出(" << out_names_.size() << "):" << std::endl;
    for (size_t i = 0; i < out_names_.size(); ++i) {
        auto sh = session_->GetOutputTypeInfo(i)
                            .GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "    " << out_names_[i] << ": [";
        for (size_t j = 0; j < sh.size(); ++j)
            std::cout << sh[j] << (j+1<sh.size()?",":"");
        std::cout << "]" << std::endl;
    }
    std::cout << "─────────────────────────────\n" << std::endl;
}

InferenceStats RangeNetInferencer::avgStats() const {
    std::lock_guard<std::mutex> lk(stat_mu_);
    if (total_frames_ == 0) return {};
    InferenceStats avg;
    const double n = static_cast<double>(total_frames_);
    avg.preprocess_ms  = accum_.preprocess_ms  / n;
    avg.inference_ms   = accum_.inference_ms   / n;
    avg.postprocess_ms = accum_.postprocess_ms / n;
    avg.total_ms       = accum_.total_ms       / n;
    return avg;
}

bool RangeNetInferencer::savePLY(const std::vector<float>&   pts,
                                  const std::vector<int32_t>& labels,
                                  const std::string&          path)
{
    const size_t n = labels.size();
    if (pts.size() / 4 != n) {
        std::cerr << "[RangeNet] savePLY: pts/labels 数量不匹配" << std::endl;
        return false;
    }
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[RangeNet] savePLY: 无法创建 " << path << std::endl;
        return false;
    }
    f << "ply\nformat ascii 1.0\n"
      << "element vertex " << n << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "property int label\nend_header\n";
    for (size_t i = 0; i < n; ++i) {
        int lb = std::max(0, std::min(NUM_CLASSES-1, (int)labels[i]));
        f << std::fixed << std::setprecision(4)
          << pts[i*4] << ' ' << pts[i*4+1] << ' ' << pts[i*4+2] << ' '
          << (int)CLASS_COLORS[lb][0] << ' '
          << (int)CLASS_COLORS[lb][1] << ' '
          << (int)CLASS_COLORS[lb][2] << ' '
          << lb << '\n';
    }
    return true;
}

} // namespace rangenet
