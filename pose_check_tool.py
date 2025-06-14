
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

st.title("人体構造の違和感検出ツール（ラフイラスト向け）")
uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return math.degrees(angle_rad)

def check_limb(name, p1, joint, p2, head_ref, max_len_ratio, angle_range, image_cv, w, h, issues):
    """Check a limb for abnormal length or bending."""

    dist = get_distance(p1, joint) + get_distance(joint, p2)
    ang = get_angle(p1, joint, p2)

    if dist > head_ref * max_len_ratio:
        cv2.line(image_cv, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (0, 0, 255), 2)
        cv2.putText(image_cv, f"{name} too long", (int(p2.x * w), int(p2.y * h) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        issues.append(f"{name} is too long")

    if ang < angle_range[0] or ang > angle_range[1]:
        px, py = int(joint.x * w), int(joint.y * h)
        cv2.putText(image_cv, f"{name} angle: {int(ang)}", (px, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(image_cv, (px, py), 6, (0, 0, 255), -1)
        issues.append(f"{name} angle out of range: {int(ang)}°")

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = pose.process(image_rgb)
    issues = []

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = image_np.shape

        head_top = landmarks[mp_pose.PoseLandmark.NOSE]
        chin = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
        head_height = get_distance(head_top, chin)

        check_limb("Left Arm",
                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
                   head_height, 3.0, (30, 180), image_np, w, h, issues)

        check_limb("Right Arm",
                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
                   head_height, 3.0, (30, 180), image_np, w, h, issues)

        check_limb("Left Leg",
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE],
                   head_height, 4.0, (30, 180), image_np, w, h, issues)

        check_limb("Right Leg",
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE],
                   head_height, 4.0, (30, 180), image_np, w, h, issues)

        st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption="判定結果", use_column_width=True)

        if issues:
            st.markdown("### 指摘内容")
            for msg in issues:
                st.write(f"- {msg}")
        else:
            st.success("大きな問題は見つかりませんでした。")
    else:
        st.warning("人物を検出できませんでした。イラストのポーズや解像度を確認してください。")
