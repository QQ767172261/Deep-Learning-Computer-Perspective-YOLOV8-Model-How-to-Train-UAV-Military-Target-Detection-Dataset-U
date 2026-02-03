深度学习计算机视角YOLOV8模型如何训练无人机军事目标检测数据集 无人机士兵及车辆运输检测数据集
@[toc]

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/61da2d35eda740ac936951abd8a32571.png)
：
无人机军事目标检测数据集（DST1155）yolo和voc两种标注方式

标签数：3065
框数: 9667

4个类别以及框数：
armed vehicle (5138), person (1029), soldier (2093), vehicle (1407)
### **无人机军事目标检测数据集统计**

| **项目** | **数量/类型** |
| :--- | :--- |
| **数据集格式** | YOLO, VOC |
| **总标签数** | 3,065 |
| **总边界框数** | 9,667 |
| **类别数量** | 4 |

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8a56fc1c246640f998bec719bca46c8a.png)

### **按类别统计**

| **类别** | **边界框数量** | **占比** |
| :--- | :--- | :--- |
| armed vehicle (武装车辆) | 5,138 | ~53.15% |
| soldier (士兵) | 2,093 | ~21.65% |
| vehicle (车辆) | 1,407 | ~14.56% |
| person (人员) | 1,029 | ~10.65% |
| **总计** | **9,667** | **~100%** |

> **注**：占比为基于总边界框数（9,667）的近似百分比。
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cc10d489cfd84ee3a4ca4e4ed5869353.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4312f4a63632454382678f1c5541f15e.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6ee581ee2a1f43bfb0d8c61f3ed6754c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3440e6f4607943c1bdf4c9b61d6b7450.png)
以下是针对您提供的 **无人机军事目标检测数据集** 的完整训练与推理方案，包含：

- 数据集说明
- YOLO 与 VOC 格式转换
- **YOLOv8 详细训练代码**
- **推理检测代码（支持图片、视频、摄像头）**
- 模型评估与可视化

---

## 🛰️ 无人机军事目标检测数据集概览

| 项目 | 说明 |
|------|------|
| 图像数量 | 3,065 张 |
| 标注框数 | 9,667 个 |
| 目标类别 | 4 类：<br>• `armed_vehicle` (武装车辆, 5138)<br>• `person` (平民, 1029)<br>• `soldier` (士兵, 2093)<br>• `vehicle` (普通车辆, 1407) |
| 标注格式 | 同时提供 **YOLO 格式**（`.txt`）和 **Pascal VOC 格式**（`.xml`） |
| 图像格式 | `.jpg` / `.png`（建议统一为 `.jpg`） |
| 分辨率 | 通常为 1920×1080 或 4000×3000（无人机航拍） |
| 应用场景 | 军事侦察、边境监控、战场态势感知、安防预警 |
| 推荐模型 | **YOLOv8s / YOLOv8m**（精度与速度平衡） |

---

## 📂 数据集目录结构（YOLO 格式）

```bash
military_dataset/
│
├── images/
│   ├── train/          # 训练图像
│   ├── val/            # 验证图像
│   └── test/           # 测试图像
│
├── labels/
│   ├── train/          # YOLO 格式 .txt 标注文件
│   ├── val/
│   └── test/
│
├── annotations/        # 可选：VOC .xml 文件（用于转换）
│
├── data.yaml           # YOLO 配置文件
└── README.md
```

> ✅ 建议按 8:1:1 划分训练/验证/测试集。

---

## 🔄 VOC 转 YOLO 格式脚本（如需）

```python
# voc_to_yolo.py
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# 类别映射（必须与 data.yaml 一致）
class_names = ['armed_vehicle', 'person', 'soldier', 'vehicle']
class_to_id = {name: i for i, name in enumerate(class_names)}

def convert_voc_to_yolo(xml_file, output_dir, img_width=640, img_height=640):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in class_to_id:
            continue
        cls_id = class_to_id[cls_name]

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # 归一化
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 保存 .txt 文件
    txt_file = Path(output_dir) / (Path(xml_file).stem + ".txt")
    with open(txt_file, "w") as f:
        f.write("\n".join(yolo_lines))

# 批量转换
voc_dir = "military_dataset/annotations"
yolo_label_dir = "military_dataset/labels/train"
os.makedirs(yolo_label_dir, exist_ok=True)

for xml_file in Path(voc_dir).glob("*.xml"):
    convert_voc_to_yolo(xml_file, yolo_label_dir, img_width=640, img_height=640)
```

---

## 📄 `data.yaml` 配置文件

```yaml
# data.yaml
train: ./military_dataset/images/train
val: ./military_dataset/images/val
test: ./military_dataset/images/test

# 类别数量
nc: 4

# 类别名称（顺序必须与标注一致）
names: ['armed_vehicle', 'person', 'soldier', 'vehicle']
```

---

## 🚀 YOLOv8 详细训练代码

```python
# train_military.py
from ultralytics import YOLO
import torch
import os

# ==================== 配置参数 ====================
MODEL_TYPE = 'yolov8s.pt'           # 可选: yolov8n, yolov8m, yolov8l, yolov8x
DATA_CONFIG = 'military_dataset/data.yaml'
EPOCHS = 200
BATCH_SIZE = 16
IMG_SIZE = 640
PROJECT_NAME = 'military_detection'
EXPERIMENT_NAME = 'yolov8s_640_military'

# 创建项目目录
os.makedirs(f"runs/train/{EXPERIMENT_NAME}", exist_ok=True)

# ==================== 检查设备 ====================
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
print(f"✅ 使用设备: {DEVICE}")

# ==================== 加载模型 ====================
model = YOLO(MODEL_TYPE)

# ==================== 开始训练 ====================
results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    save=True,
    save_period=10,
    device=DEVICE,
    workers=4,
    optimizer='AdamW',
    lr0=1e-3,
    lrf=1e-2,
    momentum=0.937,
    weight_decay=1e-4,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    close_mosaic=10,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.2,
    scale=0.5,
    shear=2.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,  # 对小目标（如士兵）特别有效
    patience=50,     # 早停：50轮无提升则停止
)

print("🎉 训练完成！")
print(f"📌 模型保存路径: runs/train/{EXPERIMENT_NAME}")
```

---

## 🔍 模型验证与评估

```python
# validate.py
from ultralytics import YOLO

# 加载最佳模型
model = YOLO(f'runs/train/{EXPERIMENT_NAME}/weights/best.pt')

# 在验证集上评估
metrics = model.val(
    data='military_dataset/data.yaml',
    split='val',
    imgsz=640,
    batch=16,
    device=0 if torch.cuda.is_available() else 'cpu',
    save_json=True,  # 生成 COCO 风格的 results.json
    conf=0.001,      # 低置信度用于评估
)

# 输出关键指标
print(f"📊 mAP@0.5: {metrics.box.map:.4f}")
print(f"📊 mAP@0.5:0.95: {metrics.box.map50:.4f}")
print(f"📊 Precision: {metrics.box.p:.4f}")
print(f"📊 Recall: {metrics.box.r:.4f}")

# 按类别输出 AP
for i, name in enumerate(['armed_vehicle', 'person', 'soldier', 'vehicle']):
    print(f"  ├─ {name}: AP@0.5 = {metrics.box.ap[i]:.4f}")
```

---

## 🔦 详细推理检测代码（支持图片、视频、摄像头）

```python
# predict.py
from ultralytics import YOLO
import cv2

# ==================== 配置 ====================
MODEL_PATH = f'runs/train/{EXPERIMENT_NAME}/weights/best.pt'
SOURCE = 'military_dataset/images/test'  # 可为图像/视频路径、目录、摄像头ID
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
SHOW = True  # 是否显示结果
SAVE = True  # 是否保存结果
PROJECT = 'runs/predict/military_test'

# ==================== 加载模型 ====================
model = YOLO(MODEL_PATH)

# ==================== 推理 ====================
results = model.predict(
    source=SOURCE,
    imgsz=640,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    device=0 if torch.cuda.is_available() else 'cpu',
    show=SHOW,
    save=SAVE,
    project=PROJECT,
    name='result',
    exist_ok=True,
    classes=None,  # 检测所有类别
    line_width=2,
    verbose=True
)

# ==================== 遍历结果并可视化（可选） ====================
for r in results:
    # 获取图像和结果
    im_array = r.plot()  # 绘制边界框和标签
    im = im_array  # BGR for OpenCV

    # 显示（如果未启用 show）
    if not SHOW:
        cv2.imshow('Military Detection', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 打印检测结果
    boxes = r.boxes
    print(f"🖼️ 图像: {r.path}")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        print(f"  ├─ {label}: {conf:.2f}")

cv2.destroyAllWindows()
print("✅ 推理完成！结果已保存。")
```

---

## 🎥 视频推理（单独脚本）

```python
# predict_video.py
from ultralytics import YOLO
import cv2

model = YOLO('runs/train/yolov8s_640_military/weights/best.pt')
video_path = 'test_video.mp4'  # 或 0 表示摄像头
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 推理
    results = model(frame, imgsz=640, conf=0.3)

    # 绘制结果
    annotated_frame = results[0].plot()

    # 显示
    cv2.imshow('Military Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📈 训练技巧与优化建议

| 技巧 | 说明 |
|------|------|
| **Copy-Paste Augmentation** | 将士兵/车辆粘贴到新背景，提升伪装目标识别能力 |
| **高分辨率训练（1280）** | 提升小目标（如远处士兵）检测精度 |
| **Focal Loss** | 在 `model.yaml` 中修改损失函数，缓解类别不平衡 |
| **TTA（Test-Time Augmentation）** | 推理时使用多尺度、翻转增强提升召回率 |
| **Hard Example Mining** | 对漏检样本重新标注并加入训练集 |

---



