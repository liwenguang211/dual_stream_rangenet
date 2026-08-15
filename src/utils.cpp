// ============================================================
// utils.cpp — 文件 I/O、评估、批量处理工具
// ============================================================
#include "RangeNetInferencer.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <sys/stat.h>
#include <dirent.h>

namespace rangenet {
namespace utils {

// ────────────────────────────────────────────────────────────
// 文件 I/O
// ────────────────────────────────────────────────────────────

/**
 * 加载 KITTI 格式 .bin 文件
 * 格式: [x, y, z, intensity] × N, 每字段 float32
 * @return 展开的 float 数组, 长度 N*4
 */
std::vector<float> loadKITTIBin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("无法打开文件: " + path);

    const std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<float> data(sz / sizeof(float));
    if (!f.read(reinterpret_cast<char*>(data.data()), sz))
        throw std::runtime_error("读取失败: " + path);
    return data;
}

/**
 * 保存标签 (int32 binary)
 */
bool saveLabels(const std::vector<int32_t>& labels,
                const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.write(reinterpret_cast<const char*>(labels.data()),
            labels.size() * sizeof(int32_t));
    return f.good();
}

// ────────────────────────────────────────────────────────────
// 统计打印
// ────────────────────────────────────────────────────────────

void printStats(const InferenceStats& s, int frame_id) {
    std::cout << "\n─── 推理统计";
    if (frame_id >= 0) std::cout << " (帧 " << frame_id << ")";
    std::cout << " ───\n"
              << std::fixed << std::setprecision(2)
              << "  点数:     " << s.num_points     << "\n"
              << "  预处理:   " << s.preprocess_ms  << " ms\n"
              << "  推理:     " << s.inference_ms   << " ms\n"  // 修正: 原版误打 preprocess_ms
              << "  后处理:   " << s.postprocess_ms << " ms\n"
              << "  合计:     " << s.total_ms       << " ms"
              << "  (FPS=" << std::setprecision(1)
              << 1000.0 / std::max(s.total_ms, 0.001) << ")\n"
              << "  置信度:   " << std::setprecision(4)
              << s.confidence_mean << "\n";
}

// ────────────────────────────────────────────────────────────
// 评估指标 (mIoU, per-class IoU)
// ────────────────────────────────────────────────────────────

void evaluate(const std::vector<int32_t>& pred,
              const std::vector<int32_t>& gt) {
    if (pred.size() != gt.size()) {
        std::cerr << "[eval] 预测/真值数量不一致" << std::endl;
        return;
    }
    // 混淆矩阵
    std::vector<std::vector<int64_t>> cm(
        NUM_CLASSES, std::vector<int64_t>(NUM_CLASSES, 0));

    for (size_t i = 0; i < pred.size(); ++i) {
        int p = std::max(0, std::min(NUM_CLASSES-1, (int)pred[i]));
        int g = std::max(0, std::min(NUM_CLASSES-1, (int)gt  [i]));
        cm[g][p]++;
    }

    // per-class IoU
    double miou = 0; int valid = 0;
    std::cout << "\n─── 各类 IoU ───\n";
    for (int c = 0; c < NUM_CLASSES; ++c) {
        int64_t tp = cm[c][c], fp = 0, fn = 0;
        for (int j = 0; j < NUM_CLASSES; ++j) {
            if (j != c) { fp += cm[j][c]; fn += cm[c][j]; }
        }
        int64_t total_gt = 0;
        for (int j = 0; j < NUM_CLASSES; ++j) total_gt += cm[c][j];
        if (total_gt == 0) continue;

        double iou = (tp + fp + fn > 0) ?
            static_cast<double>(tp) / (tp + fp + fn) : 0.0;
        miou += iou; ++valid;

        std::cout << "  " << std::setw(15) << std::left << CLASS_NAMES[c]
                  << ": " << std::fixed << std::setprecision(4) << iou
                  << "  (" << total_gt << " pts)\n";
    }
    if (valid > 0)
        std::cout << "  mIoU: " << miou / valid << "\n";
}

// ────────────────────────────────────────────────────────────
// 批量文件夹处理
// ────────────────────────────────────────────────────────────

void processFolder(RangeNetInferencer&  net,
                   const std::string&   in_dir,
                   const std::string&   out_dir) {
    // 创建输出目录
    mkdir(out_dir.c_str(), 0755);

    // 枚举 .bin 文件
    std::vector<std::string> files;
    DIR* dir = opendir(in_dir.c_str());
    if (!dir) { std::cerr << "无法打开目录: " << in_dir << std::endl; return; }
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        std::string fn = ent->d_name;
        if (fn.size() > 4 && fn.substr(fn.size()-4) == ".bin")
            files.push_back(fn);
    }
    closedir(dir);
    std::sort(files.begin(), files.end());

    std::cout << "找到 " << files.size() << " 个 .bin 文件\n";

    double total_ms = 0;
    for (size_t fi = 0; fi < files.size(); ++fi) {
        const std::string in_path  = in_dir  + "/" + files[fi];
        const std::string stem     = files[fi].substr(0, files[fi].size()-4);
        const std::string lbl_path = out_dir + "/" + stem + ".label";
        const std::string ply_path = out_dir + "/" + stem + ".ply";

        try {
            auto pts = loadKITTIBin(in_path);
            auto res = net.infer(pts);

            saveLabels(res.labels, lbl_path);
            RangeNetInferencer::savePLY(pts, res.labels, ply_path);
            printStats(res.stats, static_cast<int>(fi));
            total_ms += res.stats.total_ms;
        } catch (const std::exception& e) {
            std::cerr << "处理 " << files[fi] << " 失败: " << e.what() << std::endl;
        }
    }

    std::cout << "\n处理完成: " << files.size() << " 帧  "
              << "平均 " << (files.empty() ? 0 : total_ms / files.size())
              << " ms/帧  "
              << "FPS=" << (total_ms > 0 ? files.size() * 1000.0 / total_ms : 0)
              << std::endl;
}

} // namespace utils
} // namespace rangenet
