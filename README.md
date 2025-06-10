# Pose Check Tool

This is a Streamlit app to analyze pose structure in anime-style illustrations. It uses MediaPipe to detect human keypoints and marks anatomically incorrect poses with red highlights. Detected issues are also listed as text feedback.

## Run locally

```bash
pip install -r requirements.txt
streamlit run pose_check_tool.py
```

## Features

- Detects incorrect arm/leg lengths
- Detects joint angles (elbows, knees)
- Marks abnormal areas on the image
- Lists detected issues as text feedback
