#pragma once
// ============================================================
// RangeNetInferencer.hpp
// Dual-Stream RangeNet Lite — C++ 推理引擎
//
// 处理流程:
//   PointCloud (N×4)
//     → spherical projection → RangeImage [1,8,H,W] CHW
//     → z-score normalize
//     → ONNX Runtime inference
//     → argmax over classes → pixel labels [H,W]
//     → back-project → point labels [N]
//     → KNN fill unmapped points
// ============================================================
#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <mutex>
#include <thread>
#include <queue>
#include <functional>
#include <condition_variable>
#include <atomic>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>
#include "model_config.h"

namespace rangenet {

// ────────────────────────────────────────────────────────────
// 数据结构
// ────────────────────────────────────────────────────────────

/** 推理耗时与质量统计 */
struct InferenceStats {
    double preprocess_ms   = 0.0;
    double inference_ms    = 0.0;
    double postprocess_ms  = 0.0;
    double total_ms        = 0.0;
    int    num_points      = 0;
    double confidence_mean = 0.0;
};

/** 单帧推理结果 */
struct InferResult {
    std::vector<int32_t> labels;        // [N] 每点语义标签
    std::vector<float>   confidences;   // [N] softmax 最大概率
    InferenceStats       stats;
};

// ────────────────────────────────────────────────────────────
// 球面投影处理器
// ────────────────────────────────────────────────────────────
class RangeImageProcessor {
public:
    RangeImageProcessor(int H, int W,
                        float fov_up_deg, float fov_down_deg);

    /**
     * 点云 → CHW 范围图像 + z-score 归一化
     * @param pts  展开点云 [x0,y0,z0,i0, x1,...], 长度 N*4
     * @param img  输出 [INPUT_CHANNELS * H * W], **CHW** layout
     * @param idx  输出 [H * W], 每像素对应点索引(-1=空)
     */
    void project(const std::vector<float>& pts,
                 std::vector<float>&       img,
                 std::vector<int32_t>&     idx) const;

    /**
     * 像素级标签回投到点云
     * @param pix_labels [H*W] 像素标签
     * @param pix_conf   [H*W] 置信度
     * @param idx        [H*W] 像素→点索引
     * @param n_pts      点总数
     * @param labels     输出 [N]
     * @param confs      输出 [N]
     */
    void backProject(const std::vector<int32_t>& pix_labels,
                     const std::vector<float>&   pix_conf,
                     const std::vector<int32_t>& idx,
                     size_t                      n_pts,
                     std::vector<int32_t>&       labels,
                     std::vector<float>&         confs) const;

    /**
     * KNN 填充未被投影覆盖的点
     * 找 K 个最近已标记邻居，多数投票决定标签
     */
    void knnFill(const std::vector<float>&   pts,
                 std::vector<int32_t>&       labels,
                 std::vector<float>&         confs,
                 int                         K = KNN_K_NEIGHBORS) const;

    int H() const { return H_; }
    int W() const { return W_; }
    void setMaxRange(float r) { max_range_ = r; }

private:
    int   H_, W_;
    float fov_up_;    // radians
    float fov_down_;  // radians
    float fov_;       // |up| + |down|
    float max_range_; // 运行时最大测距（米），由 setMaxRange() 设置
};

// ────────────────────────────────────────────────────────────
// 主推理类
// ────────────────────────────────────────────────────────────
class RangeNetInferencer {
public:
    /**
     * @param model_path  ONNX 模型路径
     * @param use_gpu     CUDA provider
     * @param device_id   CUDA 设备编号
     * @param use_trt     TensorRT provider
     * @param fp16        TensorRT FP16
     * @param n_threads   CPU 内部线程数
     * @param fov_up_deg  传感器垂直 FOV 上限（度），默认 Mid-360
     * @param fov_down_deg 传感器垂直 FOV 下限（度），默认 Mid-360
     * @param max_range   最大有效测距（米），默认 Mid-360
     * @param n_keyframe_accum 关键帧叠加帧数（机械式雷达填 1）
     *
     * 速腾 / Velodyne 快速接入示例：
     *   // 速腾 RS-Helios 32线
     *   RangeNetInferencer net(model, false,0,false,false,4, 15.f,-15.f,30.f,1);
     *   // Velodyne HDL-64
     *   RangeNetInferencer net(model, false,0,false,false,4, 2.f,-24.8f,50.f,1);
     *   // 或使用预设名称（推荐）：
     *   auto* p = findSensorPreset("rs-helios");
     *   RangeNetInferencer net(model,false,0,false,false,4,
     *                          p->fov_up_deg, p->fov_down_deg,
     *                          p->max_range,  p->n_keyframe_accum);
     */
    explicit RangeNetInferencer(const std::string& model_path,
                                bool  use_gpu          = false,
                                int   device_id        = 0,
                                bool  use_trt          = false,
                                bool  fp16             = false,
                                int   n_threads        = DEFAULT_INTRA_OP_THREADS,
                                float fov_up_deg       = FOV_UP_DEG,
                                float fov_down_deg     = FOV_DOWN_DEG,
                                float max_range        = MAX_RANGE,
                                int   n_keyframe_accum = N_KEYFRAME_ACCUM);

    ~RangeNetInferencer() = default;
    RangeNetInferencer(const RangeNetInferencer&)            = delete;
    RangeNetInferencer& operator=(const RangeNetInferencer&) = delete;

    /** 加载模型 + 预热，失败时抛出异常 */
    void initialize();

    /**
     * 推理单帧
     * @param pts  展开点云, 每点 [x,y,z,intensity], 长度 N*4
     */
    InferResult infer(const std::vector<float>& pts);

    /** 批量推理 */
    std::vector<InferResult>
    inferBatch(const std::vector<std::vector<float>>& batch);

    void printModelInfo() const;
    InferenceStats avgStats() const;

    /** 保存彩色 PLY (RGB 按类别着色) */
    static bool savePLY(const std::vector<float>&   pts,
                        const std::vector<int32_t>& labels,
                        const std::string&          path);

private:
    void buildSession();
    void warmup();

    // 预处理: pts → img (CHW) + idx
    void buildRangeImage(const std::vector<float>& pts,
                         std::vector<float>&       img,
                         std::vector<int32_t>&     idx) const;

    // 运行 ORT 推理, 返回 pixel labels [H*W] 和 confidences [H*W]
    std::pair<std::vector<int32_t>, std::vector<float>>
    runORT(const std::vector<float>& img);

    // ── ORT 对象 ─────────────────────────────────────────────
    Ort::Env                      env_;
    Ort::SessionOptions           opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::RunOptions               run_opts_;

    // ── 模型 I/O 名称 ────────────────────────────────────────
    std::string              model_path_;
    std::vector<std::string> in_names_;
    std::vector<std::string> out_names_;
    std::vector<const char*> in_cstr_;    // 缓存 c_str()，避免悬空
    std::vector<const char*> out_cstr_;

    // ── 预分配缓冲 ───────────────────────────────────────────
    std::vector<float>   img_buf_;   // [8 * H * W]
    std::vector<int32_t> idx_buf_;   // [H * W]

    // ── 处理器 ───────────────────────────────────────────────
    std::unique_ptr<RangeImageProcessor> proc_;

    // ── 配置 ─────────────────────────────────────────────────
    bool  use_gpu_,  use_trt_, fp16_;
    int   device_id_, n_threads_;
    float fov_up_deg_, fov_down_deg_, max_range_;
    int   n_keyframe_accum_;

    // ── 累计统计 ─────────────────────────────────────────────
    mutable std::mutex     stat_mu_;
    mutable InferenceStats accum_;
    size_t                 total_frames_{0};
};

} // namespace rangenet
