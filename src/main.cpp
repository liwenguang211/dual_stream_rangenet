// ============================================================
// main.cpp — Dual-Stream RangeNet Lite 推理入口
//
// 用法:
//   rangenet_inference [选项]
//
//   --model  <path>   ONNX 模型路径
//   --input  <path>   .bin 文件或目录
//   --output <path>   输出目录 (默认 ./output)
//   --sensor <name>   传感器预设 (默认 mid360)
//                       mid360 | rs-helios | rs-helios16p | rs-ruby |
//                       rs-lidar16 | vlp16 | hdl32 | hdl64 | vls128
//   --fov-up <deg>    自定义垂直 FOV 上限 (覆盖 --sensor)
//   --fov-down <deg>  自定义垂直 FOV 下限 (覆盖 --sensor)
//   --max-range <m>   自定义最大测距 (覆盖 --sensor)
//   --cpu             仅 CPU 推理
//   --trt             TensorRT 加速 (需 CUDA)
//   --fp16            TRT FP16 精度
//   --threads <N>     CPU 线程数 (默认 4)
//   --benchmark       性能基准测试
//   --no-knn          禁用 KNN 标签填充
// ============================================================
#include "RangeNetInferencer.hpp"
#include "model_config.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <sys/stat.h>

using namespace rangenet;

// 前置声明 (utils.cpp)
namespace rangenet { namespace utils {
    std::vector<float> loadKITTIBin(const std::string&);
    bool               saveLabels  (const std::vector<int32_t>&, const std::string&);
    void               printStats  (const InferenceStats&, int);
    void               processFolder(RangeNetInferencer&, const std::string&, const std::string&);
}}

// ────────────────────────────────────────────────────────────
// 工具
// ────────────────────────────────────────────────────────────
static bool isFile(const std::string& p) {
    struct stat s{}; stat(p.c_str(), &s); return S_ISREG(s.st_mode);
}
static bool isDir(const std::string& p) {
    struct stat s{}; stat(p.c_str(), &s); return S_ISDIR(s.st_mode);
}

// ────────────────────────────────────────────────────────────
// 基准测试
// ────────────────────────────────────────────────────────────
static void runBenchmark(RangeNetInferencer& net,
                         size_t n_pts     = 65536,
                         int    n_warmup  = 10,
                         int    n_iters   = 100) {
    std::cout << "\n=== 性能基准测试 ===\n"
              << "点数: " << n_pts << "  迭代: " << n_iters << std::endl;

    // 生成均匀随机点云 (模拟室外场景)
    std::vector<float> pts;
    pts.reserve(n_pts * 4);
    std::srand(42);
    for (size_t i = 0; i < n_pts; ++i) {
        float ang = (float)std::rand() / RAND_MAX * 2 * 3.14159f;
        float r   = 2.f + (float)std::rand() / RAND_MAX * 60.f;
        pts.push_back(r * std::cos(ang));
        pts.push_back(r * std::sin(ang));
        pts.push_back(-1.5f + (float)std::rand() / RAND_MAX * 5.f);
        pts.push_back((float)std::rand() / RAND_MAX);
    }

    // 预热 (已在 initialize() 内部预热，这里额外跑几次)
    for (int i = 0; i < n_warmup; ++i) net.infer(pts);

    // 计时
    double total = 0;
    double min_ms = 1e9, max_ms = 0;
    for (int i = 0; i < n_iters; ++i) {
        auto res = net.infer(pts);
        total  += res.stats.total_ms;
        min_ms  = std::min(min_ms, res.stats.total_ms);
        max_ms  = std::max(max_ms, res.stats.total_ms);
    }
    double avg = total / n_iters;

    std::cout << std::fixed
              << "  avg:  " << avg   << " ms  (FPS=" << 1000.0/avg   << ")\n"
              << "  min:  " << min_ms << " ms\n"
              << "  max:  " << max_ms << " ms\n"
              << "  pts/s: " << n_pts * 1000.0 / avg << "\n";

    // 各阶段明细
    auto as = net.avgStats();
    std::cout << "\n  预处理平均:  " << as.preprocess_ms  << " ms\n"
              << "  推理平均:    " << as.inference_ms   << " ms\n"
              << "  后处理平均:  " << as.postprocess_ms << " ms\n";
}

// ────────────────────────────────────────────────────────────
// main
// ────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string model_path = MODEL_PATH;
    std::string input_path;
    std::string output_dir = "./output";
    std::string sensor_name = "mid360";
    bool use_gpu   = false;
    bool use_trt   = false;
    bool fp16      = false;
    bool benchmark = false;
    int  n_threads = DEFAULT_INTRA_OP_THREADS;
    // 自定义 FOV（-1 表示未设置，使用预设值）
    float custom_fov_up   = -1.f;
    float custom_fov_down = -1.f;
    float custom_max_range = -1.f;

    // ── 参数解析 ─────────────────────────────────────────────
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"     && i+1<argc) { model_path    = argv[++i]; }
        else if (a == "--input"     && i+1<argc) { input_path    = argv[++i]; }
        else if (a == "--output"    && i+1<argc) { output_dir    = argv[++i]; }
        else if (a == "--sensor"    && i+1<argc) { sensor_name   = argv[++i]; }
        else if (a == "--fov-up"    && i+1<argc) { custom_fov_up    = std::stof(argv[++i]); }
        else if (a == "--fov-down"  && i+1<argc) { custom_fov_down  = std::stof(argv[++i]); }
        else if (a == "--max-range" && i+1<argc) { custom_max_range = std::stof(argv[++i]); }
        else if (a == "--threads"   && i+1<argc) { n_threads     = std::atoi(argv[++i]); }
        else if (a == "--gpu")      use_gpu   = true;
        else if (a == "--trt")    { use_trt   = true; use_gpu = true; }
        else if (a == "--fp16")     fp16      = true;
        else if (a == "--cpu")      use_gpu   = false;
        else if (a == "--benchmark") benchmark = true;
        else if (a == "--help") {
            std::cout <<
R"(Dual-Stream RangeNet Lite 推理工具

用法: rangenet_inference [选项]

选项:
  --model  <path>   ONNX 模型路径 (默认: models/dual_rangenet_lite.onnx)
  --input  <path>   输入 .bin 文件 或 目录 (含多帧 .bin)
  --output <path>   输出目录 (默认: ./output)
  --sensor <name>   传感器预设 (默认: mid360)
                      mid360       Livox Mid-360      FOV -7~+52°  40m
                      rs-helios    速腾 RS-Helios 32线 FOV -15~+15° 30m
                      rs-helios16p 速腾 RS-Helios-16P  FOV -15~+15° 20m
                      rs-ruby      速腾 RS-Ruby 128线  FOV -25~+15° 30m
                      rs-lidar16   速腾 RS-LiDAR-16    FOV -15~+15° 20m
                      vlp16        Velodyne VLP-16      FOV -15~+15° 20m
                      hdl32        Velodyne HDL-32E     FOV -30~+10° 30m
                      hdl64        Velodyne HDL-64E     FOV -24~+2°  50m
                      vls128       Velodyne VLS-128     FOV -25~+15° 50m
  --fov-up <deg>    自定义垂直 FOV 上限（度，覆盖 --sensor）
  --fov-down <deg>  自定义垂直 FOV 下限（度，覆盖 --sensor）
  --max-range <m>   自定义最大测距（米，覆盖 --sensor）
  --gpu             CUDA GPU 加速
  --trt             TensorRT 加速 (需 GPU)
  --fp16            TensorRT FP16 精度
  --threads <N>     CPU 内部线程数 (默认 4)
  --benchmark       性能基准测试 (100 帧)
  --help            显示此帮助

类别索引:
  0:background  1:ground  2:roof  3:side_facade  4:front_facade
  5:beam        6:column  7:window  8:dynamic

输出文件:
  <stem>.label  — int32 二进制标签
  <stem>.ply    — RGB 着色点云 (CloudCompare 可直接打开)
)";
            return 0;
        }
    }

    try {
        // ── 解析传感器预设 ────────────────────────────────────
        const SensorPreset* preset = findSensorPreset(sensor_name);
        if (!preset) {
            std::cerr << "[错误] 未知传感器预设: " << sensor_name
                      << "\n可用预设: mid360 rs-helios rs-helios16p rs-ruby "
                         "rs-lidar16 vlp16 hdl32 hdl64 vls128" << std::endl;
            return 1;
        }
        // 自定义值覆盖预设
        float fov_up    = (custom_fov_up    > -0.5f) ? custom_fov_up    : preset->fov_up_deg;
        float fov_down  = (custom_fov_down  < 0.5f && custom_fov_down > -999.f)
                              ? custom_fov_down : preset->fov_down_deg;
        float max_range = (custom_max_range > 0.f)   ? custom_max_range  : preset->max_range;
        int   accum     = preset->n_keyframe_accum;

        // ── 创建推理器 ────────────────────────────────────────
        RangeNetInferencer net(model_path, use_gpu, 0, use_trt, fp16, n_threads,
                               fov_up, fov_down, max_range, accum);
        net.initialize();

        // ── 基准测试模式 ─────────────────────────────────────
        if (benchmark) {
            runBenchmark(net);
            return 0;
        }

        // ── 创建输出目录 ─────────────────────────────────────
        mkdir(output_dir.c_str(), 0755);

        // ── 输入处理 ─────────────────────────────────────────
        if (input_path.empty()) {
            std::cout << "\n[提示] 未指定 --input，运行示例推理\n";
            // 生成简单示例点云 (地面平面 + 一面墙)
            std::vector<float> pts;
            for (int i = 0; i < 30000; ++i) {   // 地面
                float x = -30 + (float)std::rand()/RAND_MAX*60;
                float y = -30 + (float)std::rand()/RAND_MAX*60;
                pts.insert(pts.end(), {x, y, -0.5f, 0.3f});
            }
            for (int i = 0; i < 10000; ++i) {   // 立面
                float x = 20.f + (float)std::rand()/RAND_MAX*2;
                float y = -15 + (float)std::rand()/RAND_MAX*30;
                float z = -1  + (float)std::rand()/RAND_MAX*8;
                pts.insert(pts.end(), {x, y, z, 0.5f});
            }

            auto res = net.infer(pts);
            utils::printStats(res.stats, 0);

            // 打印类别分布
            std::vector<int> cnt(NUM_CLASSES, 0);
            for (int lb : res.labels) cnt[std::max(0,std::min(NUM_CLASSES-1,lb))]++;
            std::cout << "\n类别分布:\n";
            for (int c = 0; c < NUM_CLASSES; ++c)
                if (cnt[c] > 0)
                    std::cout << "  " << CLASS_NAMES[c] << ": " << cnt[c] << "\n";

            RangeNetInferencer::savePLY(pts, res.labels,
                                        output_dir + "/example.ply");
            std::cout << "\n保存: " << output_dir << "/example.ply\n";

        } else if (isFile(input_path)) {
            // ── 单帧处理 ─────────────────────────────────────
            auto pts = utils::loadKITTIBin(input_path);
            auto res = net.infer(pts);
            utils::printStats(res.stats, 0);

            // 提取文件名
            size_t sl = input_path.find_last_of("/\\");
            std::string stem = input_path.substr(sl+1);
            stem = stem.substr(0, stem.rfind('.'));

            utils::saveLabels(res.labels,   output_dir + "/" + stem + ".label");
            RangeNetInferencer::savePLY(pts, res.labels, output_dir + "/" + stem + ".ply");

            std::cout << "\n保存:\n"
                      << "  " << output_dir << "/" << stem << ".label\n"
                      << "  " << output_dir << "/" << stem << ".ply\n";

        } else if (isDir(input_path)) {
            // ── 批量处理目录 ─────────────────────────────────
            utils::processFolder(net, input_path, output_dir);

        } else {
            std::cerr << "输入路径不存在: " << input_path << std::endl;
            return 1;
        }

        // ── 最终统计 ─────────────────────────────────────────
        auto as = net.avgStats();
        if (as.total_ms > 0) {
            std::cout << "\n=== 总体平均性能 ===\n"
                      << "  预处理: " << as.preprocess_ms  << " ms\n"
                      << "  推理:   " << as.inference_ms   << " ms\n"
                      << "  后处理: " << as.postprocess_ms << " ms\n"
                      << "  合计:   " << as.total_ms       << " ms"
                      << "  (FPS=" << 1000.0/as.total_ms << ")\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "[错误] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
