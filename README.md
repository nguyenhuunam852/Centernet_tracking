# 🚀 Centernet_tracking

## Yo, What's This All About? 😎

Welcome to **Centernet_tracking**, the coolest project for smashing object detection and tracking like a pro!  
Built on the epic **CenterNet** architecture and enhanced with **DeepSORT**, this bad boy lets you detect and track objects in videos or images with ninja-level precision.

Whether you're diving into computer vision for fun, building the next big thing in surveillance, or creating something wild for autonomous systems — this repo is your ticket to greatness!

---

## 💥 Why You'll Love It

- 🔍 **CenterNet Detection**: Fast and accurate object detection using keypoint-based approach  
- 🧠 **DeepSORT Tracking**: ID-aware tracking with ReID feature embedding for better identity consistency  
- 🎯 **Track Like a Boss**: Follow multiple objects across frames, even after occlusion  
- 🎨 **Make It Your Own**: Easily configurable and modular for your project needs  
- ⚡ **Ready-to-Roll Models**: Pre-trained models to get you started in a snap

---

## 🛠️ What You Need to Get Started

Make sure you’ve got the following:

- 🐍 Python 3.8+  
- 🔥 PyTorch 1.7.0+  
- 🎥 OpenCV, NumPy, SciPy  
- ⚡ CUDA (optional but recommended)  
- 🧬 TensorFlow (for DeepSORT ReID model)  

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🎉 Let’s Get It Running!

### 1. Clone the Repo

```bash
git clone https://github.com/nguyenhuunam852/Centernet_tracking.git
cd Centernet_tracking
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Download Pretrained Models

```bash
# For CenterNet
wget <centernet_model_url> -P models/

# For DeepSORT (ReID model)
wget <deepsort_reid_model_url> -P models/reid/
```

---

## ✨ How to Make Magic Happen

### 1. Prepare Your Data

Place your input videos or images in the `data/` folder.  
Supported formats: MP4, JPG, PNG, etc.

### 2. Run Detection + DeepSORT Tracking

```bash
python main.py --mode inference --input_path data/your_video.mp4 --output_path output/ --tracker deepsort
```

### 3. Train CenterNet (Optional)

```bash
python main.py --mode train --config configs/default.yaml --data_path data/your_dataset/
```

### 4. Customize Settings

Edit `configs/default.yaml` to tweak:

- Model type (CenterNet variant)
- Input size
- Tracking algorithm: `deepsort`, `kalman`, or `iou`
- ReID model path
- Batch size, learning rate, etc.

---

## 🎯 About DeepSORT Integration

This project integrates **DeepSORT** for improved object tracking performance.  
Unlike basic IOU/Kalman tracking, DeepSORT uses ReID embeddings to help maintain consistent identities across occlusions and long durations.

### DeepSORT Pipeline:

- Appearance embeddings using a ReID model (usually ResNet-based)
- Kalman filter + Hungarian algorithm for motion + visual matching
- ID assignment that resists ID-switching

> 📁 Place your ReID model in: `models/reid/`

---

## 🔥 Try This Out!

```bash
python main.py --mode inference --input_path data/sample_video.mp4 --output_path output/tracked_sample.mp4 --tracker deepsort
```

---

## 🎥 What You’ll Get

- Output video with bounding boxes + tracking IDs
- Results saved to the `output/` folder
- Logs/statistics optionally printed or saved

---

## 📦 Pre-trained Models

Pre-trained models for detection and DeepSORT coming soon — stay tuned!  
Place `.pth` and `.ckpt` files into:

```
models/
├── centernet.pth
└── reid/
    └── deepsort.ckpt
```

---

## 🙌 Contributing

Want to improve this project?

1. Fork the repo  
2. Create a branch: `git checkout -b my-feature`  
3. Commit changes: `git commit -m "Added awesome stuff"`  
4. Push it: `git push origin my-feature`  
5. Submit a pull request 🚀

---

## 📄 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for more details.

---

## 🌟 Acknowledgments

- 📌 [CenterNet: Objects as Points](https://arxiv.org/abs/1904.07850)  
- 📌 [DeepSORT](https://arxiv.org/abs/1703.07402)  
- 🙏 Thanks to the open-source community for making this possible!

---

## 💬 Contact

Have a question or suggestion?  
Open an issue or reach out at [github.com/nguyenhuunam852](https://github.com/nguyenhuunam852)

Let’s build something legendary together! ✨
