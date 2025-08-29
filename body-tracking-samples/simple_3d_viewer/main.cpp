// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <iostream>
#include <map>
#include <vector>
#include <k4arecord/playback.h>
#include <k4a/k4a.h>
#include <k4abt.h>
#include <eigen3/Eigen/Dense>
#include <fstream>

#include <BodyTrackingHelpers.h>
#include <Utilities.h>
#include <Window3dWrapper.h>
#include <lcm/lcm-cpp.hpp>
#include "lcm/skeleton_lcm.hpp"
#include "lcm/multi_sparse_cloud_t.hpp"
#include <opencv2/opencv.hpp>

lcm::LCM skeleton_lcm_pub("udpm://239.255.76.67:7667?ttl=255");
k4a_calibration_t sensorCalibration;

void PrintUsage()
{
#ifdef _WIN32
    printf("\nUSAGE: (k4abt_)simple_3d_viewer.exe SensorMode[NFOV_UNBINNED, WFOV_BINNED](optional) RuntimeMode[CPU, CUDA, DIRECTML, TENSORRT](optional) -model MODEL_PATH(optional)\n");
#else
    printf("\nUSAGE: (k4abt_)simple_3d_viewer.exe SensorMode[NFOV_UNBINNED, WFOV_BINNED](optional) RuntimeMode[CPU, CUDA, TENSORRT](optional)\n");
#endif
    printf("  - SensorMode: \n");
    printf("      NFOV_UNBINNED (default) - Narrow Field of View Unbinned Mode [Resolution: 640x576; FOI: 75 degree x 65 degree]\n");
    printf("      WFOV_BINNED             - Wide Field of View Binned Mode [Resolution: 512x512; FOI: 120 degree x 120 degree]\n");
    printf("  - RuntimeMode: \n");
    printf("      CPU - Use the CPU only mode. It runs on machines without a GPU but it will be much slower\n");
    printf("      CUDA - Use CUDA for processing.\n");
#ifdef _WIN32
    printf("      DIRECTML - Use the DirectML processing mode.\n");
#endif
    printf("      TENSORRT - Use the TensorRT processing mode.\n");
    printf("      OFFLINE - Play a specified file. Does not require Kinect device\n");
    printf("e.g.   (k4abt_)simple_3d_viewer.exe WFOV_BINNED CPU\n");
    printf("e.g.   (k4abt_)simple_3d_viewer.exe CPU\n");
    printf("e.g.   (k4abt_)simple_3d_viewer.exe WFOV_BINNED\n");
    printf("e.g.   (k4abt_)simple_3d_viewer.exe OFFLINE MyFile.mkv\n");
}

void PrintAppUsage()
{
    printf("\n");
    printf(" Basic Navigation:\n\n");
    printf(" Rotate: Rotate the camera by moving the mouse while holding mouse left button\n");
    printf(" Pan: Translate the scene by holding Ctrl key and drag the scene with mouse left button\n");
    printf(" Zoom in/out: Move closer/farther away from the scene center by scrolling the mouse scroll wheel\n");
    printf(" Select Center: Center the scene based on a detected joint by right clicking the joint with mouse\n");
    printf("\n");
    printf(" Key Shortcuts\n\n");
    printf(" ESC: quit\n");
    printf(" h: help\n");
    printf(" b: body visualization mode\n");
    printf(" k: 3d window layout\n");
    printf("\n");
}

const char* k4abt_joint_id_to_string(int joint) {
  static const char* joint_names[] = {
    "PELVIS",
    "SPINE_NAVEL",
    "SPINE_CHEST",
    "NECK",
    "CLAVICLE_LEFT",
    "SHOULDER_LEFT",
    "ELBOW_LEFT",
    "WRIST_LEFT",
    "HAND_LEFT",
    "HANDTIP_LEFT",
    "THUMB_LEFT",
    "CLAVICLE_RIGHT",
    "SHOULDER_RIGHT",
    "ELBOW_RIGHT",
    "WRIST_RIGHT",
    "HAND_RIGHT",
    "HANDTIP_RIGHT",
    "THUMB_RIGHT",
    "HIP_LEFT",
    "KNEE_LEFT",
    "ANKLE_LEFT",
    "FOOT_LEFT",
    "HIP_RIGHT",
    "KNEE_RIGHT",
    "ANKLE_RIGHT",
    "FOOT_RIGHT",
    "HEAD",
    "NOSE",
    "EYE_LEFT",
    "EAR_LEFT",
    "EYE_RIGHT",
    "EAR_RIGHT"
  };

  if (joint >= 0 && joint < K4ABT_JOINT_COUNT)
    return joint_names[joint];
  else
    return "UNKNOWN_JOINT";
}

// Global State and Key Process Function
bool s_isRunning = true;
Visualization::Layout3d s_layoutMode = Visualization::Layout3d::OnlyMainView;
bool s_visualizeJointFrame = false;


int64_t ProcessKey(void* /*context*/, int key)
{
    // https://www.glfw.org/docs/latest/group__keys.html
    switch (key)
    {
        // Quit
    case GLFW_KEY_ESCAPE:
        s_isRunning = false;
        break;
    case GLFW_KEY_K:
        s_layoutMode = (Visualization::Layout3d)(((int)s_layoutMode + 1) % (int)Visualization::Layout3d::Count);
        break;
    case GLFW_KEY_B:
        s_visualizeJointFrame = !s_visualizeJointFrame;
        break;
    case GLFW_KEY_H:
        PrintAppUsage();
        break;
    }
    return 1;
}

int64_t CloseCallback(void* /*context*/)
{
    s_isRunning = false;
    return 1;
}

struct InputSettings
{
    k4a_depth_mode_t DepthCameraMode = K4A_DEPTH_MODE_NFOV_UNBINNED;
#ifdef _WIN32
    k4abt_tracker_processing_mode_t processingMode = K4ABT_TRACKER_PROCESSING_MODE_GPU_DIRECTML;
#else
    k4abt_tracker_processing_mode_t processingMode = K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA;
#endif
    bool Offline = false;
    std::string FileName;
    std::string ModelPath;
};

struct BoneLabel2DConfig {
    float max_pixel_dist = 5.0f;  // ignore pixels farther than this (px) from any bone
    float z_weight       = 0.10f;  // weight for depth-consistency term (mm^2 -> px^2 scale)
    bool  use_z_check    = true;   // add Z penalty to break 2D overlaps
    bool  skip_low_conf  = true;   // ignore bones whose joints are below LOW confidence
    int   stride         = 10;    // 10×10 ⇒ ~1/100 pixels
    bool  fill_blocks    = true;  // replicate label into the stride block for display
};
// 2D point-to-segment squared distance, also returns t in [0,1]
static inline float pointSegDist2_2D(const cv::Point2f& p,
                                     const cv::Point2f& a,
                                     const cv::Point2f& b,
                                     float* t_out=nullptr)
{
    cv::Point2f ab = b - a;
    float denom = std::max(1e-12f, ab.dot(ab));
    float t = ((p - a).dot(ab)) / denom;
    t = std::min(1.f, std::max(0.f, t));
    if (t_out) *t_out = t;
    cv::Point2f proj = a + t * ab;
    cv::Point2f d = p - proj;
    return d.dot(d);
}

// 1) Distinct colors per GROUP (HSV → BGR)
static std::vector<cv::Scalar> makeColors(size_t n) {
    std::vector<cv::Scalar> out; out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        int H = int(180.0 * i / std::max<size_t>(1, n));
        cv::Mat hsv(1,1,CV_8UC3, cv::Scalar(H, 200, 255));
        cv::Mat bgr; cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cv::Vec3b b = bgr.at<cv::Vec3b>(0,0);
        out.emplace_back(b[0], b[1], b[2]); // B,G,R
    }
    return out;
}


struct BoneKey {
    int a, b;
    bool operator==(const BoneKey& o) const { return a==o.a && b==o.b; }
};
struct BoneKeyHash {
    size_t operator()(const BoneKey& k) const {
        return (size_t)k.a * 1315423911u ^ (size_t)k.b;
    }
};

static inline float clamp01(float t){ return t<0.f?0.f:(t>1.f?1.f:t); }

struct SparseCloudCfg {
    int   stride         = 10;    // 10x10 → ~1% samples
    float max_pixel_dist = 10.0f; // px
    float z_weight       = 0.10f; // (mm)^2 scale
    bool  use_z_check    = true;
    bool  skip_low_conf  = true;
    bool  include_gid0   = true;  // false to drop ungrouped
    const char* frame_id = "depth";
};
struct BodyProj {
    std::array<cv::Point2f, K4ABT_JOINT_COUNT> uv;
    std::array<uint8_t,    K4ABT_JOINT_COUNT> ok;
    std::array<float,      K4ABT_JOINT_COUNT> z_mm; // depth-cam Z of joints (mm)
    // k4abt_skeleton_t* skel = nullptr; // only for confidence checking
    std::array<uint8_t,    K4ABT_JOINT_COUNT> conf{};
};

// ---------- 2) Define gid → color here (MANUAL PALETTE) ----------
// BGR order!  (cv::Scalar(Blue, Green, Red))
static std::unordered_map<int, cv::Scalar> makeGroupColorMap()
{
    std::unordered_map<int, cv::Scalar> C;

    // 0 = ungrouped (anything not in makeGroupMap)
    C[0]  = cv::Scalar(80, 80, 80);       // neutral gray

    // Example colors — change as you like:
    C[1]  = cv::Scalar(0, 165, 255);      // trunk: orange
    C[2]  = cv::Scalar(0, 255, 255);      // neck: yellow
    C[3]  = cv::Scalar(255, 0, 255);      // clavicle L: magenta
    C[4]  = cv::Scalar(255, 0, 128);      // upper arm L: purple-ish
    C[5]  = cv::Scalar(255, 0, 180);      // forearm L: pink-ish

    // (you commented-out some left-hand groups; they will fall back to gid 0)

    C[6]  = cv::Scalar(255, 255, 0);        // L hip: green
    C[7]  = cv::Scalar(60, 255, 60);      // L upper leg
    C[8]  = cv::Scalar(100, 200, 100);    // L lower leg
    C[9]  = cv::Scalar(120, 180, 120);    // L foot

    C[10] = cv::Scalar(255, 255, 0);      // head: cyan?

    C[11] = cv::Scalar(0, 0, 255);        // clavicle R: red
    C[12] = cv::Scalar(0, 64, 255);       // upper arm R: deep red
    C[13] = cv::Scalar(0, 128, 255);      // forearm R: orange-red
    // (you commented-out R hand groups; they’ll be gid 0)

    C[14] = cv::Scalar(0, 200, 0);        // R hip
    C[15] = cv::Scalar(40, 220, 40);      // R upper leg
    C[16] = cv::Scalar(80, 240, 80);      // R lower leg
    C[17] = cv::Scalar(120, 255, 120);    // R foot

    return C;
}

// ---------- 3) Fallback color if a gid has no manual entry ----------
static cv::Scalar fallbackColorForGroup(int gid)
{
    // HSV → BGR based on gid (stable, distinct)
    int H = (gid * 37) % 180; // hue in [0,179]
    cv::Mat hsv(1,1,CV_8UC3, cv::Scalar(H, 200, 255));
    cv::Mat bgr; cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b b = bgr.at<cv::Vec3b>(0,0);
    return cv::Scalar(b[0], b[1], b[2]); // B,G,R
}


static std::unordered_map<BoneKey,int,BoneKeyHash> makeGroupMap() {
    std::unordered_map<BoneKey,int,BoneKeyHash> G;

    auto add = [&](int gid, k4abt_joint_id_t A, k4abt_joint_id_t B){
        G[{(int)A,(int)B}] = gid;
    };

    int gid = 1;
    // Group 1: trunk
    add(gid, K4ABT_JOINT_SPINE_CHEST, K4ABT_JOINT_SPINE_NAVEL);
    add(gid, K4ABT_JOINT_SPINE_NAVEL, K4ABT_JOINT_PELVIS);
    add(gid, K4ABT_JOINT_SPINE_CHEST, K4ABT_JOINT_NECK);
    add(gid, K4ABT_JOINT_SPINE_CHEST, K4ABT_JOINT_CLAVICLE_LEFT);
    add(gid, K4ABT_JOINT_SPINE_CHEST, K4ABT_JOINT_CLAVICLE_RIGHT);
    ++gid;

    // Group 2: neck
    add(gid, K4ABT_JOINT_NECK, K4ABT_JOINT_HEAD);
    // todo fix this, this is supposed to be head
    add(gid, K4ABT_JOINT_HEAD, K4ABT_JOINT_NOSE);
    ++gid;

    // Group 3: clavicle L
    add(gid, K4ABT_JOINT_CLAVICLE_LEFT, K4ABT_JOINT_SHOULDER_LEFT); ++gid;

    // Group 4: upper arm L
    add(gid, K4ABT_JOINT_SHOULDER_LEFT, K4ABT_JOINT_ELBOW_LEFT); ++gid;

    // Group 5: forearm L
    add(gid, K4ABT_JOINT_ELBOW_LEFT, K4ABT_JOINT_WRIST_LEFT); ++gid;

    // Group 0: hand L
    // add(gid, K4ABT_JOINT_WRIST_LEFT, K4ABT_JOINT_HAND_LEFT); ++gid;

    // Group 0: fingertips/thumb L
    // add(gid, K4ABT_JOINT_HAND_LEFT,  K4ABT_JOINT_HANDTIP_LEFT);
    // add(gid, K4ABT_JOINT_WRIST_LEFT, K4ABT_JOINT_THUMB_LEFT);
    // ++gid;

    // Group 6: left hip
    add(gid, K4ABT_JOINT_PELVIS,     K4ABT_JOINT_HIP_LEFT); ++gid;
    // Group 7: left upper leg
    add(gid, K4ABT_JOINT_HIP_LEFT,   K4ABT_JOINT_KNEE_LEFT); ++gid;
    // Group 8: left lower leg
    add(gid, K4ABT_JOINT_KNEE_LEFT,  K4ABT_JOINT_ANKLE_LEFT); ++gid;
    // Group 9: left foot
    add(gid, K4ABT_JOINT_ANKLE_LEFT, K4ABT_JOINT_FOOT_LEFT); ++gid;

    // Group 10: head lines
    add(gid, K4ABT_JOINT_NOSE,      K4ABT_JOINT_EYE_LEFT);
    add(gid, K4ABT_JOINT_EYE_LEFT,  K4ABT_JOINT_EAR_LEFT);
    add(gid, K4ABT_JOINT_NOSE,      K4ABT_JOINT_EYE_RIGHT);
    add(gid, K4ABT_JOINT_EYE_RIGHT, K4ABT_JOINT_EAR_RIGHT);
    ++gid;

    // Group 11: right clavicle
    add(gid, K4ABT_JOINT_CLAVICLE_RIGHT, K4ABT_JOINT_SHOULDER_RIGHT); ++gid;
    // Group 12: right upper arm
    add(gid, K4ABT_JOINT_SHOULDER_RIGHT, K4ABT_JOINT_ELBOW_RIGHT); ++gid;
    // Group 13: right forearm
    add(gid, K4ABT_JOINT_ELBOW_RIGHT,    K4ABT_JOINT_WRIST_RIGHT); ++gid;

    // Group 0: hand R
    // add(gid, K4ABT_JOINT_WRIST_RIGHT,    K4ABT_JOINT_HAND_RIGHT); ++gid;
    // add(gid, K4ABT_JOINT_HAND_RIGHT,     K4ABT_JOINT_HANDTIP_RIGHT);
    // add(gid, K4ABT_JOINT_WRIST_RIGHT,    K4ABT_JOINT_THUMB_RIGHT);
    // ++gid;

    // Group 14: right hip
    add(gid, K4ABT_JOINT_PELVIS,     K4ABT_JOINT_HIP_RIGHT); ++gid;
    // Group 15: right upper leg
    add(gid, K4ABT_JOINT_HIP_RIGHT,  K4ABT_JOINT_KNEE_RIGHT); ++gid;
    // Group 16: right lower leg
    add(gid, K4ABT_JOINT_KNEE_RIGHT, K4ABT_JOINT_ANKLE_RIGHT); ++gid;
    // Group 17: right foot
    add(gid, K4ABT_JOINT_ANKLE_RIGHT,K4ABT_JOINT_FOOT_RIGHT); ++gid;

    return G;
}

// ---------- 4) Build per-bone color table using the maps above ----------
static std::vector<cv::Scalar> makeBoneColorsFromGroups(
    const std::array<std::pair<k4abt_joint_id_t,k4abt_joint_id_t>, 31>& g_boneList)
{
    auto G = makeGroupMap();          // Bone pair → gid (your function)
    auto C = makeGroupColorMap();     // gid → color (manual)

    std::vector<cv::Scalar> boneColors(g_boneList.size());

    for (size_t b = 0; b < g_boneList.size(); ++b)
    {
        BoneKey k{ (int)g_boneList[b].first, (int)g_boneList[b].second };
        auto it = G.find(k);

        int gid = (it != G.end()) ? it->second : 0;     // 0 = ungrouped default
        auto ct = C.find(gid);
        boneColors[b] = (ct != C.end()) ? ct->second    // manual color
                                        : fallbackColorForGroup(gid); // or HSV fallback
    }
    return boneColors;
}

static std::vector<cv::Scalar> makeBoneColorsByGroup_Map(
    const std::array<std::pair<k4abt_joint_id_t,k4abt_joint_id_t>, 31>& g_boneList)
{
    auto G = makeGroupMap();

    // Count groups
    int max_gid = -1;
    std::vector<int> bone2group(g_boneList.size(), 0);
    for (size_t b = 0; b < g_boneList.size(); ++b) {
        BoneKey k{(int)g_boneList[b].first, (int)g_boneList[b].second};
        auto it = G.find(k);
        int gid = (it != G.end()) ? it->second : 0; // 0 corresponds to ungrouped
        bone2group[b] = gid;
        if (gid > max_gid) max_gid = gid;
    }
    int numGroups = max_gid + 1;

    std::vector<cv::Scalar> groupColors = makeColors(numGroups);
    std::vector<cv::Scalar> boneColors(g_boneList.size());
    for (size_t b = 0; b < g_boneList.size(); ++b) {
        boneColors[b] = groupColors[ bone2group[b] ];
    }
    return boneColors;
}
static std::vector<BodyProj> preprojectBodies(const k4abt_frame_t bodyFrame,
                                              const k4a_calibration_t& calib,
                                              int W, int H)
{
    uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
    std::vector<BodyProj> out(numBodies);
    for (uint32_t i = 0; i < numBodies; ++i) {
        k4abt_body_t body;
        if (k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton) != K4A_RESULT_SUCCEEDED)
            continue;
        // out[i].skel = &body.skeleton;
        for (int j = 0; j < (int)K4ABT_JOINT_COUNT; ++j) {
            const k4a_float3_t& P = body.skeleton.joints[j].position; // depth-cam mm
            out[i].conf[j] = body.skeleton.joints[j].confidence_level;
            out[i].z_mm[j] = P.v[2];

            k4a_float2_t p2; int valid = 0;
            k4a_calibration_3d_to_2d(&calib, &P,
                                     K4A_CALIBRATION_TYPE_DEPTH,
                                     K4A_CALIBRATION_TYPE_DEPTH,
                                     &p2, &valid);
            if (valid) {
                float u = p2.xy.x, v = p2.xy.y;
                out[i].uv[j] = cv::Point2f(u, v);
                out[i].ok[j] = (u>=0 && u<W && v>=0 && v<H);
            } else {
                out[i].ok[j] = 0; out[i].z_mm[j] = P.v[2];
            }
        }
    }
    return out;
}
static inline int boneIndexToGroupId(size_t b,
                                     const std::array<std::pair<k4abt_joint_id_t,k4abt_joint_id_t>, 31>& bones,
                                     const std::unordered_map<BoneKey,int,BoneKeyHash>& G)
{
    BoneKey k{ (int)bones[b].first, (int)bones[b].second };
    auto it = G.find(k);
    return (it != G.end()) ? it->second : 0; // 0 = ungrouped
}

// ---------- One-message publisher ----------
static void PublishSparseClassCloudsUnified(const k4abt_frame_t bodyFrame,
                                            const k4a_image_t depthImage,
                                            const k4a_image_t bodyIndexMap,
                                            const k4a_calibration_t& calib,
                                            const SparseCloudCfg& cfg)
{
    const int W = calib.depth_camera_calibration.resolution_width;
    const int H = calib.depth_camera_calibration.resolution_height;

    const uint16_t* depth = reinterpret_cast<const uint16_t*>(k4a_image_get_buffer(depthImage));
    const uint8_t*  bidx  = k4a_image_get_buffer(bodyIndexMap);

    // Precompute
    auto proj = preprojectBodies(bodyFrame, calib, W, H);
    const uint32_t numBodies = (uint32_t)proj.size();
    const float max_d2 = cfg.max_pixel_dist * cfg.max_pixel_dist;
    auto G = makeGroupMap();

    // Accumulate xyz per gid
    std::unordered_map<int, std::vector<float>> class_xyz;
    class_xyz.reserve(32);
    bool early_break = false;

    // Dithered sampling to avoid fixed grid artifacts
    static int frame_shift = 0;
    frame_shift = (frame_shift + 1) % cfg.stride;
    int y0 = frame_shift, x0 = (3 * frame_shift) % cfg.stride;

    for (int y = y0; y < H; y += cfg.stride) {
        for (int x = x0; x < W; x += cfg.stride) {

            int cx = std::min(x + cfg.stride/2, W-1);
            int cy = std::min(y + cfg.stride/2, H-1);
            int idx = cy * W + cx;

            if (numBodies == 0) continue;
            uint8_t bi = bidx[idx];
            if (bi == K4ABT_BODY_INDEX_MAP_BACKGROUND || bi >= numBodies) continue;

            uint16_t z_px = depth[idx];
            if (z_px == 0) continue;

            const BodyProj& B = proj[bi];
            cv::Point2f p((float)cx, (float)cy);

            // nearest 2D bone (with optional Z penalty)
            int bestBone = -1; float bestScore = std::numeric_limits<float>::max();
            for (size_t b = 0; b < g_boneList.size(); ++b) {
                auto j1 = g_boneList[b].first;
                auto j2 = g_boneList[b].second;

                if (!B.ok[j1] || !B.ok[j2]) continue;
                if (cfg.skip_low_conf && (B.conf[j1] < K4ABT_JOINT_CONFIDENCE_LOW || B.conf[j2] < K4ABT_JOINT_CONFIDENCE_LOW))
                  continue;

                float t=0.f;
                float d2 = pointSegDist2_2D(p, B.uv[j1], B.uv[j2], &t);
                if (d2 > max_d2) continue;

                float score = d2;
                if (cfg.use_z_check) {
                    float z_exp = (1.f - t) * B.z_mm[j1] + t * B.z_mm[j2];
                    float z_err = float(z_px) - z_exp;
                    score += cfg.z_weight * (z_err * z_err);
                }
                if (score < bestScore) { bestScore = score; bestBone = (int)b; }
            }
            if (bestBone < 0) continue;

            int gid = boneIndexToGroupId((size_t)bestBone, g_boneList, G);
            if (!cfg.include_gid0 && gid == 0) continue;

            // 2D→3D in depth camera frame
            k4a_float2_t p2{ (float)cx, (float)cy };
            k4a_float3_t P3; int valid = 0;
            k4a_calibration_2d_to_3d(&calib, &p2, (float)z_px,
                                     K4A_CALIBRATION_TYPE_DEPTH,
                                     K4A_CALIBRATION_TYPE_DEPTH,
                                     &P3, &valid);
            if (!valid) continue;

            auto& vec = class_xyz[gid];
            vec.push_back(P3.xyz.x);
            vec.push_back(P3.xyz.y);
            vec.push_back(P3.xyz.z);
        }
    }

    // Build unified LCM message
    // Sort gids for deterministic order
    std::map<int, std::vector<float>> ordered(class_xyz.begin(), class_xyz.end());

    multi_sparse_cloud_t msg;
    msg.frame_id   = cfg.frame_id;

    auto now       = std::chrono::system_clock::now();
    msg.utime      = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    msg.num_classes = (int32_t)ordered.size();
    msg.classes.resize(msg.num_classes);

    int i = 0;
    std::cout << "ordered size: " << ordered.size() << std::endl;
    for (auto& kv : ordered) {
        int gid = kv.first;
        const auto& xyz = kv.second;
        const int n = (int)(xyz.size());
        std::cout << "id: " << gid << ", data: " << xyz[0] << ", " << xyz[1] << ", " << xyz[2] << ", num: " << n << std::endl;

        msg.classes[i].class_id = gid;
        msg.classes[i].n        = n;
        msg.classes[i].xyz      = xyz;     // std::vector<float>; must be size 3*n
        ++i;
    }

    skeleton_lcm_pub.publish("SPARSE_CLOUDS_ALL", &msg);
}
/**
 * Labels each depth pixel with the nearest bone (in 2D) of its body.
 * - depthImage: current K4A depth image (for Z penalty)
 * - bodyIndexMap: K4A body index map (size = depth)
 * - calibration: full k4a_calibration_t (for 3D->2D projection)
 * - out_bone_label: length W*H, -1 for background, otherwise bone index in g_boneList
 * - out_colors: optional per-pixel BGR colors (same size), colored by bone
 */
void LabelBonesPerPixel2D(const k4abt_frame_t bodyFrame,
                          const k4a_image_t depthImage,
                          const k4a_image_t bodyIndexMap,
                          const k4a_calibration_t& calibration,
                          const BoneLabel2DConfig& cfg,
                          std::vector<int>& out_bone_label,
                          std::vector<cv::Vec3b>* out_colors = nullptr)
{
    const int W = calibration.depth_camera_calibration.resolution_width;
    const int H = calibration.depth_camera_calibration.resolution_height;

    out_bone_label.assign(W * H, -1);
    if (out_colors) out_colors->assign(W * H, cv::Vec3b(0,0,0));

    const uint16_t* depth = reinterpret_cast<const uint16_t*>(k4a_image_get_buffer(depthImage));
    const uint8_t*  bidx  = k4a_image_get_buffer(bodyIndexMap);

    const uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
    if (numBodies == 0) return;

    // Precompute per-bone colors
    // static std::vector<cv::Scalar> BONE_COLORS = makeBoneColors(g_boneList.size());
    // ===== Use this once (after g_boneList is defined) =====
    // static std::vector<cv::Scalar> BONE_COLORS = makeBoneColorsByGroup_Map(g_boneList);
    static std::vector<cv::Scalar> BONE_COLORS = makeBoneColorsFromGroups(g_boneList);
    // Precompute, for each body index i, the 2D projections and Z of joints

    std::vector<BodyProj> proj(numBodies);

    for (uint32_t i = 0; i < numBodies; ++i) {
        k4abt_body_t body;
        if (k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton) != K4A_RESULT_SUCCEEDED) continue;
        // proj[i].skel = &body.skeleton;
        // proj[i].conf = {};

        for (int j = 0; j < (int)K4ABT_JOINT_COUNT; ++j) {
            const k4a_float3_t& P = body.skeleton.joints[j].position; // depth cam space (mm)
            k4a_float2_t p2; int valid = 0;
            k4a_calibration_3d_to_2d(&calibration, &P,
                                     K4A_CALIBRATION_TYPE_DEPTH,    // source
                                     K4A_CALIBRATION_TYPE_DEPTH,    // target plane (depth image)
                                     &p2, &valid);
            proj[i].conf[j] = body.skeleton.joints[j].confidence_level;
            if (valid) {
                float u = p2.xy.x, v = p2.xy.y;
                bool in = (u >= 0 && u < W && v >= 0 && v < H);
                proj[i].uv[j] = cv::Point2f(u, v);
                proj[i].ok[j] = in ? 1 : 0;
                proj[i].z_mm[j] = P.v[2];
            } else {
                proj[i].ok[j] = 0;
                proj[i].z_mm[j] = P.v[2];
            }
        }
    }

    const float max_d2 = cfg.max_pixel_dist * cfg.max_pixel_dist;

    for (int y = 0; y < H; y += cfg.stride) {
        for (int x = 0; x < W; x += cfg.stride) {

            // choose block “representative” pixel (center of block is nicer)
            int cx = std::min(x + cfg.stride/2, W-1);
            int cy = std::min(y + cfg.stride/2, H-1);
            int cidx = cy * W + cx;

            uint8_t bi = bidx[cidx];
            if (bi == K4ABT_BODY_INDEX_MAP_BACKGROUND || bi >= numBodies) continue;

            const uint16_t z_px = depth[cidx];
            if (z_px == 0) continue;

            const BodyProj& B = proj[bi];  // (proj[] computed earlier, unchanged)

            int bestBone = -1;
            float bestScore = std::numeric_limits<float>::max();
            const cv::Point2f p((float)cx, (float)cy);

            for (size_t b = 0; b < g_boneList.size(); ++b) {
                const auto j1 = g_boneList[b].first;
                const auto j2 = g_boneList[b].second;

                if (!B.ok[j1] || !B.ok[j2]) continue;
                if (cfg.skip_low_conf &&
                    (B.conf[j1] < K4ABT_JOINT_CONFIDENCE_LOW ||
                    B.conf[j2] < K4ABT_JOINT_CONFIDENCE_LOW))
                    continue;

                float t = 0.f;
                float d2 = pointSegDist2_2D(p, B.uv[j1], B.uv[j2], &t);
                if (d2 > max_d2) continue;

                float score = d2;
                if (cfg.use_z_check) {
                    float z_exp = (1.f - t) * B.z_mm[j1] + t * B.z_mm[j2]; // mm
                    float z_err = float(z_px) - z_exp;
                    score += cfg.z_weight * (z_err * z_err);
                }

                if (score < bestScore) { bestScore = score; bestBone = (int)b; }
            }

            if (bestBone < 0) continue;

            // Write label at the representative pixel
            out_bone_label[cidx] = bestBone;
            if (out_colors) {
                const cv::Scalar c = BONE_COLORS[bestBone];
                (*out_colors)[cidx] = cv::Vec3b((uchar)c[0], (uchar)c[1], (uchar)c[2]);
            }

            // Optionally fill the whole stride×stride block (only same-person pixels)
            if (cfg.fill_blocks) {
                const uint8_t block_body = bi;
                const cv::Scalar c = BONE_COLORS[bestBone];

                int y1 = y, y2 = std::min(y + cfg.stride, H);
                int x1 = x, x2 = std::min(x + cfg.stride, W);

                for (int yy = y1; yy < y2; ++yy) {
                    int base = yy * W;
                    for (int xx = x1; xx < x2; ++xx) {
                        int ii = base + xx;
                        if (bidx[ii] != block_body) continue;   // don’t spill across bodies
                        out_bone_label[ii] = bestBone;
                        if (out_colors) (*out_colors)[ii] = cv::Vec3b((uchar)c[0], (uchar)c[1], (uchar)c[2]);
                    }
                }
            }
        }
    }
}

bool ParseInputSettingsFromArg(int argc, char** argv, InputSettings& inputSettings)
{
    for (int i = 1; i < argc; i++)
    {
        std::string inputArg(argv[i]);
        if (inputArg == std::string("NFOV_UNBINNED"))
        {
            inputSettings.DepthCameraMode = K4A_DEPTH_MODE_NFOV_UNBINNED;
        }
        else if (inputArg == std::string("WFOV_BINNED"))
        {
            inputSettings.DepthCameraMode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
        }
        else if (inputArg == std::string("CPU"))
        {
            inputSettings.processingMode = K4ABT_TRACKER_PROCESSING_MODE_CPU;
        }
        else if (inputArg == std::string("TENSORRT"))
        {
            inputSettings.processingMode = K4ABT_TRACKER_PROCESSING_MODE_GPU_TENSORRT;
        }
        else if (inputArg == std::string("CUDA"))
        {
            inputSettings.processingMode = K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA;
        }
#ifdef _WIN32
        else if (inputArg == std::string("DIRECTML"))
        {
            inputSettings.processingMode = K4ABT_TRACKER_PROCESSING_MODE_GPU_DIRECTML;
        }
#endif
        else if (inputArg == std::string("OFFLINE"))
        {
            inputSettings.Offline = true;
            if (i < argc - 1) {
                // Take the next argument after OFFLINE as file name
                inputSettings.FileName = argv[i + 1];
                i++;
            }
            else {
                return false;
            }
        }
        else if (inputArg == std::string("-model"))
        {
            if (i < argc - 1)
                inputSettings.ModelPath = argv[++i];
            else
            {
                printf("Error: model path missing\n");
                return false;
            }
        }
        else
        {
            printf("Error: command not understood: %s\n", inputArg.c_str());
            return false;
        }
    }
    return true;
}

Eigen::Matrix3f QuaternionToRotationMatrix(const k4a_quaternion_t& q) {
  Eigen::Quaternionf quat(q.v[0], q.v[1], q.v[2], q.v[3]);  // w, x, y, z
  return quat.normalized().toRotationMatrix();
}

// static std::vector<cv::Scalar> makeBoneColors(size_t n)
// {
//     // Distinct hues in HSV → BGR
//     std::vector<cv::Scalar> out;
//     out.reserve(n);
//     for (size_t i = 0; i < n; ++i)
//     {
//         int H = int(180.0 * i / std::max<size_t>(1, n)); // [0..179]
//         cv::Mat hsv(1,1,CV_8UC3, cv::Scalar(H, 200, 255));
//         cv::Mat bgr; cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
//         cv::Vec3b b = bgr.at<cv::Vec3b>(0,0);
//         out.emplace_back(b[0], b[1], b[2]); // B,G,R
//     }
//     return out;
// }
// targetCamera: K4A_CALIBRATION_TYPE_DEPTH if drawing on depth image,
//               K4A_CALIBRATION_TYPE_COLOR if drawing on a color image.
void DrawBones2D(cv::Mat& canvasBGR,
                 k4abt_frame_t bodyFrame,
                 const k4a_calibration_t& calibration,
                 k4a_calibration_type_t targetCamera)
{
    // Prepare colors (one per bone)
    // static std::vector<cv::Scalar> BONE_COLORS = makeBoneColors(g_boneList.size());
    static std::vector<cv::Scalar> BONE_COLORS = makeBoneColorsByGroup_Map(g_boneList);

    const int W = canvasBGR.cols;
    const int H = canvasBGR.rows;

    uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);

    for (uint32_t i = 0; i < numBodies; ++i)
    {
        k4abt_body_t body;
        if (k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton) != K4A_RESULT_SUCCEEDED)
            continue;

        // Project all 3D joints (in DEPTH camera coords, mm) → 2D pixels in targetCamera
        std::array<cv::Point2f, K4ABT_JOINT_COUNT> uv;
        std::array<bool,        K4ABT_JOINT_COUNT> ok{};
        for (int j = 0; j < (int)K4ABT_JOINT_COUNT; ++j)
        {
            const k4a_float3_t& p3 = body.skeleton.joints[j].position; // depth cam space
            k4a_float2_t p2; int valid = 0;

            k4a_calibration_3d_to_2d(&calibration,
                                     &p3,
                                     K4A_CALIBRATION_TYPE_DEPTH,   // source = depth camera space
                                     targetCamera,                 // target = depth or color image plane
                                     &p2, &valid);

            if (valid)
            {
                uv[j] = cv::Point2f(p2.xy.x, p2.xy.y);
                ok[j] = (uv[j].x >= 0 && uv[j].x < W && uv[j].y >= 0 && uv[j].y < H);
            }
            else
            {
                ok[j] = false;
            }
        }

        // Draw bones
        for (size_t b = 0; b < g_boneList.size(); ++b)
        {
            const k4abt_joint_id_t j1 = g_boneList[b].first;
            const k4abt_joint_id_t j2 = g_boneList[b].second;

            // Require at least LOW confidence and valid projection for both ends
            if (body.skeleton.joints[j1].confidence_level < K4ABT_JOINT_CONFIDENCE_LOW) continue;
            if (body.skeleton.joints[j2].confidence_level < K4ABT_JOINT_CONFIDENCE_LOW) continue;
            if (!ok[j1] || !ok[j2]) continue;

            const cv::Point p1(cvRound(uv[j1].x), cvRound(uv[j1].y));
            const cv::Point p2(cvRound(uv[j2].x), cvRound(uv[j2].y));

            cv::line(canvasBGR, p1, p2, BONE_COLORS[b], /*thickness*/2, cv::LINE_AA);
            // (optional) draw joint markers with the same bone color
            cv::circle(canvasBGR, p1, 3, BONE_COLORS[b], cv::FILLED, cv::LINE_AA);
            cv::circle(canvasBGR, p2, 3, BONE_COLORS[b], cv::FILLED, cv::LINE_AA);
        }
    }
}

void VisualizeResult(k4abt_frame_t bodyFrame, Window3dWrapper& window3d, int depthWidth, int depthHeight) {

    // Obtain original capture that generates the body tracking result
    k4a_capture_t originalCapture = k4abt_frame_get_capture(bodyFrame);
    k4a_image_t depthImage = k4a_capture_get_depth_image(originalCapture);

    std::vector<Color> pointCloudColors(depthWidth * depthHeight, { 1.f, 1.f, 1.f, 1.f });

    // Read body index map and assign colors
    k4a_image_t bodyIndexMap = k4abt_frame_get_body_index_map(bodyFrame);
    const uint8_t* bodyIndexMapBuffer = k4a_image_get_buffer(bodyIndexMap);
    for (int i = 0; i < depthWidth * depthHeight; i++)
    {
        uint8_t bodyIndex = bodyIndexMapBuffer[i];
        if (bodyIndex != K4ABT_BODY_INDEX_MAP_BACKGROUND)
        {
            uint32_t bodyId = k4abt_frame_get_body_id(bodyFrame, bodyIndex);
            pointCloudColors[i] = g_bodyColors[bodyId % g_bodyColors.size()];
        }
    }
    k4a_image_release(bodyIndexMap);

    // Visualize point cloud
    window3d.UpdatePointClouds(depthImage, pointCloudColors);

    // Visualize the skeleton data
    window3d.CleanJointsAndBones();
    uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);

    std::vector<Eigen::Vector3f> axes = {
      { 1.0f, 0.0f, 0.0f },
      { 0.0f, 1.0f, 0.0f },
      { 0.0f, 0.0f, 1.0f }
    };

    // // Draw skeleton on depth image
    // cv::Mat depth16(sensorCalibration.depth_camera_calibration.resolution_height,
    //                 sensorCalibration.depth_camera_calibration.resolution_width,
    //                 CV_16U,
    //                 (void*)k4a_image_get_buffer(depthImage),
    //                 k4a_image_get_stride_bytes(depthImage));
    // // Make a BGR canvas to draw on (normalize depth for visualization)
    // cv::Mat depth8, canvasBGR;
    // double minv, maxv; cv::minMaxIdx(depth16, &minv, &maxv);
    // depth16.convertTo(depth8, CV_8U, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));
    // cv::cvtColor(depth8, canvasBGR, cv::COLOR_GRAY2BGR);
    // // Draw all bones (projected into the depth image)
    // DrawBones2D(canvasBGR, bodyFrame, sensorCalibration, K4A_CALIBRATION_TYPE_DEPTH);
    // cv::imshow("Skeleton 2D (Depth)", canvasBGR);
    // cv::waitKey(1);



    std::vector<int> boneLabel2D;
    std::vector<cv::Vec3b> boneColors2D;  // optional, for preview
    BoneLabel2DConfig cfg;                 // tweak if needed

    LabelBonesPerPixel2D(bodyFrame, depthImage, bodyIndexMap,
                        sensorCalibration, cfg,
                        boneLabel2D, &boneColors2D);

    const int W = sensorCalibration.depth_camera_calibration.resolution_width;
    const int H = sensorCalibration.depth_camera_calibration.resolution_height;
    // (A) Preview on depth image:
    cv::Mat depth16(H, W, CV_16U, (void*)k4a_image_get_buffer(depthImage), k4a_image_get_stride_bytes(depthImage));
    cv::Mat depth8, depth_bgr, bone_bgr(H, W, CV_8UC3, boneColors2D.data());
    double minv, maxv; cv::minMaxIdx(depth16, &minv, &maxv);
    depth16.convertTo(depth8, CV_8U, 255.0/(maxv - minv), -minv*255.0/(maxv - minv));
    cv::cvtColor(depth8, depth_bgr, cv::COLOR_GRAY2BGR);
    cv::Mat overlay;
    cv::addWeighted(depth_bgr, 0.5, bone_bgr, 0.5, 0.0, overlay);
    cv::imshow("Bones (2D)", overlay); cv::waitKey(1);

    // (B) Reuse labels to color the 3D point cloud you already push to Window3d
    for (int i = 0; i < W * H; ++i) {
        int b = boneLabel2D[i];
        if (b >= 0) {
            const auto& c = g_bodyColors[b % g_bodyColors.size()];
            pointCloudColors[i] = { c.r, c.g, c.b, 1.f }; // or your per-bone palette
        }
    }

    SparseCloudCfg scfg;
    PublishSparseClassCloudsUnified(bodyFrame, depthImage, bodyIndexMap, sensorCalibration, scfg);



    auto now = std::chrono::system_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    std::cout << "Current time (microseconds since epoch): " << us << std::endl;

    static std::ofstream skeleton_file("/home/s/skeleton.txt", std::ios::out);
    skeleton_file << "Timestamp: " << us << "\n";

    for (uint32_t i = 0; i < numBodies; i++)
    {
        skeleton_lcm skeletons;
        k4abt_body_t body;
        VERIFY(k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton), "Get skeleton from body frame failed!");
        body.id = k4abt_frame_get_body_id(bodyFrame, i);
        skeletons.id = body.id;

        // Assign the correct color based on the body id
        Color color = g_bodyColors[body.id % g_bodyColors.size()];
        color.a = 0.4f;
        Color lowConfidenceColor = color;
        lowConfidenceColor.a = 0.1f;

        // Visualize joints
        for (int joint = 0; joint < static_cast<int>(K4ABT_JOINT_COUNT); joint++)
        {
          // std::cout << "Joint: " << joint << ", Confidence: " << body.skeleton.joints[joint].confidence_level << std::endl;
          const k4a_float3_t& jointPosition = body.skeleton.joints[joint].position;
          const k4a_quaternion_t& jointOrientation = body.skeleton.joints[joint].orientation;
          skeletons.confidence[joint] = body.skeleton.joints[joint].confidence_level;
          skeletons.joint_positions[joint][0] = jointPosition.v[0];
          skeletons.joint_positions[joint][1] = jointPosition.v[1];
          skeletons.joint_positions[joint][2] = jointPosition.v[2];
          skeletons.joint_orientations[joint][0] = jointOrientation.v[0];
          skeletons.joint_orientations[joint][1] = jointOrientation.v[1];
          skeletons.joint_orientations[joint][2] = jointOrientation.v[2];
          skeletons.joint_orientations[joint][3] = jointOrientation.v[3];
          // skeleton_file << k4abt_joint_id_to_string(joint) << ": "
          //               << jointPosition.v[0] << ", "
          //               << jointPosition.v[1] << ", "
          //               << jointPosition.v[2] << ", "
          //               << jointOrientation.v[0] << ", "
          //               << jointOrientation.v[1] << ", "
          //               << jointOrientation.v[2] << ", "
          //               << jointOrientation.v[3] << "\n";

            if (body.skeleton.joints[joint].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW)
            {
                const k4a_float3_t& jointPosition = body.skeleton.joints[joint].position;
                const k4a_quaternion_t& jointOrientation = body.skeleton.joints[joint].orientation;

                window3d.AddJoint(
                    jointPosition,
                    jointOrientation,
                    body.skeleton.joints[joint].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM ? color : lowConfidenceColor);
            }
        }

        // Visualize bones
        for (size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++)
        {
            k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
            k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;

            if (body.skeleton.joints[joint1].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW &&
                body.skeleton.joints[joint2].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW)
            {
                bool confidentBone = body.skeleton.joints[joint1].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM &&
                    body.skeleton.joints[joint2].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM;
                const k4a_float3_t& joint1Position = body.skeleton.joints[joint1].position;
                const k4a_float3_t& joint2Position = body.skeleton.joints[joint2].position;




                const k4a_quaternion_t& joint1Orientation = body.skeleton.joints[joint1].orientation;
                const k4a_quaternion_t& joint2Orientation = body.skeleton.joints[joint2].orientation;
                Eigen::Matrix3f R = QuaternionToRotationMatrix(joint1Orientation);

                if (joint1 == K4ABT_JOINT_SHOULDER_RIGHT && joint2 == K4ABT_JOINT_ELBOW_RIGHT) {
                  Eigen::Matrix3f R2 = QuaternionToRotationMatrix(joint2Orientation);
                  Eigen::Matrix3f R_diff = R.transpose() * R2; // Calculate the relative rotation
                  Eigen::Vector3f angles = R_diff.eulerAngles(2, 1, 0) * 180.0f / M_PI; // Get the Euler angles (yaw, pitch, roll)
                  // std::cout << "Relative angle: " << angles.transpose() << std::endl;
                }

                if (joint1 == K4ABT_JOINT_ELBOW_RIGHT) {
                  Eigen::Vector3f ypr = R.eulerAngles(2, 1, 0) * 180.0f / M_PI; // Convert to degrees
                  // std::cout << "ELBOW angle: " << ypr.transpose() << std::endl;
                } else if (joint1 == K4ABT_JOINT_SHOULDER_RIGHT) {
                  Eigen::Vector3f ypr = R.eulerAngles(2, 1, 0) * 180.0f / M_PI; // Convert to degrees
                  // std::cout << "SHOULDER angle: " << ypr.transpose() << std::endl;
                } else {
                  // continue;
                }
                Eigen::Vector3f origin(joint1Position.v[0], joint1Position.v[1], joint1Position.v[2]);

                float axis_length = 50.0f; // Length of the axesc

                Eigen::Vector3f x_axis = R * axes[0] * axis_length + origin;
                Eigen::Vector3f y_axis = R * axes[1] * axis_length + origin; // origin + axis_length * R.col(1);
                Eigen::Vector3f z_axis = R * axes[2] * axis_length + origin; // origin + axis_length * R.col(2);

                window3d.AddBone(joint1Position, { x_axis[0], x_axis[1], x_axis[2] }, { 1.f, 0.f, 0.f, 1.f });  // Red: X
                window3d.AddBone(joint1Position, { y_axis[0], y_axis[1], y_axis[2] }, { 0.f, 1.f, 0.f, 1.f });  // Green: Y
                window3d.AddBone(joint1Position, { z_axis[0], z_axis[1], z_axis[2] }, { 0.f, 0.f, 1.f, 1.f });  // Blue: Z


                

                window3d.AddBone(joint1Position, joint2Position, confidentBone ? color : lowConfidenceColor);
            }
        }
      skeleton_lcm_pub.publish("skeletons", &skeletons);
    }

    k4a_capture_release(originalCapture);
    k4a_image_release(depthImage);

}

void PlayFile(InputSettings inputSettings)
{
    // Initialize the 3d window controller
    Window3dWrapper window3d;

    //create the tracker and playback handle
    k4a_calibration_t sensorCalibration;
    k4abt_tracker_t tracker = nullptr;
    k4a_playback_t playbackHandle = nullptr;

    const char* file = inputSettings.FileName.c_str();
    if (k4a_playback_open(file, &playbackHandle) != K4A_RESULT_SUCCEEDED)
    {
        printf("Failed to open recording: %s\n", file);
        return;
    }

    if (k4a_playback_get_calibration(playbackHandle, &sensorCalibration) != K4A_RESULT_SUCCEEDED)
    {
        printf("Failed to get calibration\n");
        return;
    }

    k4a_capture_t capture = nullptr;
    k4a_stream_result_t playbackResult = K4A_STREAM_RESULT_SUCCEEDED;

    k4abt_tracker_configuration_t trackerConfig = K4ABT_TRACKER_CONFIG_DEFAULT;
    trackerConfig.processing_mode = inputSettings.processingMode;
    trackerConfig.model_path = inputSettings.ModelPath.c_str();
    VERIFY(k4abt_tracker_create(&sensorCalibration, trackerConfig, &tracker), "Body tracker initialization failed!");

    int depthWidth = sensorCalibration.depth_camera_calibration.resolution_width;
    int depthHeight = sensorCalibration.depth_camera_calibration.resolution_height;

    window3d.Create("3D Visualization", sensorCalibration);
    window3d.SetCloseCallback(CloseCallback);
    window3d.SetKeyCallback(ProcessKey);

    while (playbackResult == K4A_STREAM_RESULT_SUCCEEDED && s_isRunning)
    {
        playbackResult = k4a_playback_get_next_capture(playbackHandle, &capture);
        if (playbackResult == K4A_STREAM_RESULT_EOF)
        {
            // End of file reached
            break;
        }

        if (playbackResult == K4A_STREAM_RESULT_SUCCEEDED)
        {
            // check to make sure we have a depth image
            k4a_image_t depthImage = k4a_capture_get_depth_image(capture);
            if (depthImage == nullptr) {
                //If no depth image, print a warning and skip to next frame
                std::cout << "Warning: No depth image, skipping frame!" << std::endl;
                k4a_capture_release(capture);
                continue;
            }
            // Release the Depth image
            k4a_image_release(depthImage);

            //enque capture and pop results - synchronous
            k4a_wait_result_t queueCaptureResult = k4abt_tracker_enqueue_capture(tracker, capture, K4A_WAIT_INFINITE);

            // Release the sensor capture once it is no longer needed.
            k4a_capture_release(capture);

            if (queueCaptureResult == K4A_WAIT_RESULT_FAILED)
            {
                std::cout << "Error! Add capture to tracker process queue failed!" << std::endl;
                break;
            }

            k4abt_frame_t bodyFrame = nullptr;
            k4a_wait_result_t popFrameResult = k4abt_tracker_pop_result(tracker, &bodyFrame, K4A_WAIT_INFINITE);
            if (popFrameResult == K4A_WAIT_RESULT_SUCCEEDED)
            {
                /************* Successfully get a body tracking result, process the result here ***************/
                VisualizeResult(bodyFrame, window3d, depthWidth, depthHeight); 
                //Release the bodyFrame
                k4abt_frame_release(bodyFrame);
            }
            else
            {
                std::cout << "Pop body frame result failed!" << std::endl;
                break;
            }
        }

        window3d.SetLayout3d(s_layoutMode);
        window3d.SetJointFrameVisualization(s_visualizeJointFrame);
        window3d.Render();
    }

    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);
    window3d.Delete();
    printf("Finished body tracking processing!\n");
    k4a_playback_close(playbackHandle);
}

void PlayFromDevice(InputSettings inputSettings) 
{
    k4a_device_t device = nullptr;
    VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");

    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.depth_mode = inputSettings.DepthCameraMode;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras failed!");

    // Get calibration information
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensorCalibration),
        "Get depth camera calibration failed!");
    int depthWidth = sensorCalibration.depth_camera_calibration.resolution_width;
    int depthHeight = sensorCalibration.depth_camera_calibration.resolution_height;
    // std::cout << "Depth Camera Resolution: " << depthWidth << " x " << depthHeight << std::endl;

    // k4a_calibration_camera_t depth_calib = sensorCalibration.depth_camera_calibration;
    // std::cout << "Kinect Depth Camera Intrinsic Parameters:" << std::endl;
    // std::cout << "Resolution: " << depth_calib.resolution_width << " x " << depth_calib.resolution_height << std::endl;
    // std::cout << "fx: " << depth_calib.intrinsics.parameters.param.fx << std::endl;
    // std::cout << "fy: " << depth_calib.intrinsics.parameters.param.fy << std::endl;
    // std::cout << "cx: " << depth_calib.intrinsics.parameters.param.cx << std::endl;
    // std::cout << "cy: " << depth_calib.intrinsics.parameters.param.cy << std::endl;
    // std::cout << "k1: " << depth_calib.intrinsics.parameters.param.k1 << std::endl;
    // std::cout << "k2: " << depth_calib.intrinsics.parameters.param.k2 << std::endl;
    // std::cout << "k3: " << depth_calib.intrinsics.parameters.param.k3 << std::endl;
    // std::cout << "k4: " << depth_calib.intrinsics.parameters.param.k4 << std::endl;
    // std::cout << "k5: " << depth_calib.intrinsics.parameters.param.k5 << std::endl;
    // std::cout << "k6: " << depth_calib.intrinsics.parameters.param.k6 << std::endl;
    // std::cout << "codx: " << depth_calib.intrinsics.parameters.param.codx << std::endl;
    // std::cout << "cody: " << depth_calib.intrinsics.parameters.param.cody << std::endl;
    // std::cout << "p2: " << depth_calib.intrinsics.parameters.param.p2 << std::endl;
    // std::cout << "p1: " << depth_calib.intrinsics.parameters.param.p1 << std::endl;
    // std::cout << "metric_radius: " << depth_calib.intrinsics.parameters.param.metric_radius << std::endl;

    // k4a_calibration_camera_t rgb_calib = sensorCalibration.color_camera_calibration;
    // std::cout << "Kinect RGB Camera Intrinsic Parameters:" << std::endl;
    // std::cout << "Resolution: " << rgb_calib.resolution_width << " x " << rgb_calib.resolution_height << std::endl;
    // std::cout << "fx: " << rgb_calib.intrinsics.parameters.param.fx << std::endl;
    // std::cout << "fy: " << rgb_calib.intrinsics.parameters.param.fy << std::endl;
    // std::cout << "cx: " << rgb_calib.intrinsics.parameters.param.cx << std::endl;
    // std::cout << "cy: " << rgb_calib.intrinsics.parameters.param.cy << std::endl;
    // std::cout << "k1: " << rgb_calib.intrinsics.parameters.param.k1 << std::endl;
    // std::cout << "k2: " << rgb_calib.intrinsics.parameters.param.k2 << std::endl;
    // std::cout << "k3: " << rgb_calib.intrinsics.parameters.param.k3 << std::endl;
    // std::cout << "k4: " << rgb_calib.intrinsics.parameters.param.k4 << std::endl;
    // std::cout << "k5: " << rgb_calib.intrinsics.parameters.param.k5 << std::endl;
    // std::cout << "k6: " << rgb_calib.intrinsics.parameters.param.k6 << std::endl;
    // std::cout << "codx: " << rgb_calib.intrinsics.parameters.param.codx << std::endl;
    // std::cout << "cody: " << rgb_calib.intrinsics.parameters.param.cody << std::endl;
    // std::cout << "p2: " << rgb_calib.intrinsics.parameters.param.p2 << std::endl;
    // std::cout << "p1: " << rgb_calib.intrinsics.parameters.param.p1 << std::endl;
    // std::cout << "metric_radius: " << rgb_calib.intrinsics.parameters.param.metric_radius << std::endl;
    // std::cout << "fx: " << rgb_calib.intrinsics.parameters.v[0] << std::endl;

    // Create Body Tracker
    k4abt_tracker_t tracker = nullptr;
    k4abt_tracker_configuration_t trackerConfig = K4ABT_TRACKER_CONFIG_DEFAULT;
    trackerConfig.processing_mode = inputSettings.processingMode;
    trackerConfig.model_path = inputSettings.ModelPath.c_str();
    VERIFY(k4abt_tracker_create(&sensorCalibration, trackerConfig, &tracker), "Body tracker initialization failed!");

    // Initialize the 3d window controller
    Window3dWrapper window3d;
    window3d.Create("3D Visualization", sensorCalibration);
    window3d.SetCloseCallback(CloseCallback);
    window3d.SetKeyCallback(ProcessKey);

    while (s_isRunning)
    {
        k4a_capture_t sensorCapture = nullptr;
        k4a_wait_result_t getCaptureResult = k4a_device_get_capture(device, &sensorCapture, 0); // timeout_in_ms is set to 0

        if (getCaptureResult == K4A_WAIT_RESULT_SUCCEEDED)
        {
            // timeout_in_ms is set to 0. Return immediately no matter whether the sensorCapture is successfully added
            // to the queue or not.
            k4a_wait_result_t queueCaptureResult = k4abt_tracker_enqueue_capture(tracker, sensorCapture, 0);

            // Release the sensor capture once it is no longer needed.
            k4a_capture_release(sensorCapture);

            if (queueCaptureResult == K4A_WAIT_RESULT_FAILED)
            {
                std::cout << "Error! Add capture to tracker process queue failed!" << std::endl;
                break;
            }
        }
        else if (getCaptureResult != K4A_WAIT_RESULT_TIMEOUT)
        {
            std::cout << "Get depth capture returned error: " << getCaptureResult << std::endl;
            break;
        }

        // Pop Result from Body Tracker
        k4abt_frame_t bodyFrame = nullptr;
        k4a_wait_result_t popFrameResult = k4abt_tracker_pop_result(tracker, &bodyFrame, 0); // timeout_in_ms is set to 0
        if (popFrameResult == K4A_WAIT_RESULT_SUCCEEDED)
        {
            /************* Successfully get a body tracking result, process the result here ***************/
            VisualizeResult(bodyFrame, window3d, depthWidth, depthHeight);
            //Release the bodyFrame
            k4abt_frame_release(bodyFrame);
        }
       
        window3d.SetLayout3d(s_layoutMode);
        window3d.SetJointFrameVisualization(s_visualizeJointFrame);
        window3d.Render();
    }

    std::cout << "Finished body tracking processing!" << std::endl;

    window3d.Delete();
    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);

    k4a_device_stop_cameras(device);
    k4a_device_close(device);
}

int main(int argc, char** argv)
{
    InputSettings inputSettings;
   
    if (!ParseInputSettingsFromArg(argc, argv, inputSettings))
    {
        // Print app usage if user entered incorrect arguments.
        PrintUsage();
        return -1;
    }

    // Either play the offline file or play from the device
    if (inputSettings.Offline == true)
    {
        PlayFile(inputSettings);
    }
    else
    {
        PlayFromDevice(inputSettings);
    }

    return 0;
}
