# 🚀 Centernet_tracking

## Yo, What's This All About? 😎

Welcome to **Centernet_tracking**, the coolest project for smashing object detection and tracking like a pro!  
Built on the epic **CenterNet** architecture, this bad boy lets you detect and track objects in videos or images with ninja-level precision.

Whether you're diving into computer vision for fun, building the next big thing in surveillance, or creating something wild for autonomous systems — this repo is your ticket to greatness!

---

## 💥 Why You'll Love It

- 🔥 **Next-Level Detection**: CenterNet brings the heat with fast and accurate object spotting.  
- 🎯 **Track Like a Boss**: Follow multiple objects across frames like they're your crew.  
- 🎨 **Make It Your Own**: Tweak it, twist it, make it fit your vibe!  
- ⚡ **Ready-to-Roll Models**: Pre-trained models to get you started in a snap.

---

## 🛠️ What You Need to Get Started

Make sure you’ve got the following:

- 🐍 Python 3.8+ (because we’re modern like that)  
- 🔥 PyTorch 1.7.0+ (the engine of our dreams)  
- 🧠 OpenCV, NumPy, SciPy (the sidekicks)  
- ⚡ CUDA (optional, for that GPU turbo boost)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🎉 Let’s Get It Running!

### 1. Grab the Code

```bash
git clone https://github.com/nguyenhuunam852/Centernet_tracking.git
cd Centernet_tracking
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Download Pretrained Models (optional)

```bash
wget <model_url> -P models/
```

---

## ✨ How to Make Magic Happen

### 1. Prep Your Data

Place your images or videos in the `data/` folder.  
Supported formats: COCO, MOT, or custom (check `configs/` for settings).

### 2. Track Like a Superstar

Run detection and tracking on a video or image set:

```bash
python main.py --mode inference --input_path data/your_video.mp4 --output_path output/
```

### 3. Train Your Own Beast

Train the model on your own dataset:

```bash
python main.py --mode train --config configs/default.yaml --data_path data/your_dataset/
```

### 4. Customize Your Ride

Edit `configs/default.yaml` to:

- Pick your model type  
- Set input resolution  
- Choose tracking algorithm (`IOU`, `Kalman`, etc.)  
- Adjust batch size, learning rate, and more

---

## 🔥 Try This Out!

```bash
# Track objects in a sample video
python main.py --mode inference --input_path data/sample_video.mp4 --output_path output/tracked_video.mp4
```

---

## 🎥 What You’ll Get

Your tracked videos or annotated images will land in the `output/` folder,  
complete with slick bounding boxes and tracking IDs.

Show off your masterpiece! 🧠💻

---

## 📦 Pre-trained Models

Pre-trained models coming soon — stay tuned!  
Place downloaded weights into the `models/` folder.

---

## 🙌 Join the Squad!

Wanna make this project even more awesome? Here’s how to contribute:

1. Fork this repo like it’s hot 🔥  
2. Create a new branch:  
   ```bash
   git checkout -b my-cool-feature
   ```
3. Commit your changes:  
   ```bash
   git commit -m "Added some epic sauce"
   ```
4. Push it up:  
   ```bash
   git push origin my-cool-feature
   ```
5. Open a Pull Request. Let’s make waves together 🌊

---

## 📄 License

This project is rocking the **MIT License**.  
See the `LICENSE` file for all the legal stuff.

---

## 🌟 Big Thanks!

- Shoutout to **CenterNet: Objects as Points** for the inspo.  
- Big love to the open-source community for tools & datasets that power this project.

---

## 💬 Got Questions? Let’s Chat!

Open an issue on GitHub or DM [nguyenhuunam852](https://github.com/nguyenhuunam852).  
Let’s build something legendary together! 🚀
