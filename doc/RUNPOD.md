# ⚡ SwiftEdit — Hướng dẫn chạy trên RunPod

> **Yêu cầu GPU tối thiểu:** VRAM ≥ 24 GB  
> Đã kiểm thử trên: A100 40 GB · RTX 3090 · RTX 4090 · A5000 · A6000

---

## 1. Tạo Pod trên RunPod

1. Đăng nhập [runpod.io](https://www.runpod.io) → **Pods** → **+ Deploy**.
2. Chọn GPU **VRAM ≥ 24 GB** (ví dụ RTX 4090 hoặc A100).
3. Chọn **Template**: `RunPod PyTorch 2.x` (Ubuntu 22.04, Python 3.10+).
4. Mở mục **Expose HTTP Ports** → thêm port `7860` (Gradio UI).
5. Mục **Network Volume** (khuyến nghị):
   - Tạo hoặc gắn một **Network Volume ≥ 50 GB** (dùng để cache HuggingFace models).
   - Gắn vào mount point `/workspace/data`.
6. Bấm **Deploy** và đợi Pod khởi động.

---

## 2. Cài đặt môi trường

Mở **Web Terminal** hoặc kết nối SSH vào Pod:

```bash
# Clone repo (branch exp_v2)
cd /workspace
git clone https://github.com/paingoat/SwiftEdit.git
cd SwiftEdit
git switch exp_v3

# Cài đặt thư viện Python
# (requirements.txt đã bao gồm PyTorch nightly + CUDA 12.8)
pip install -r requirements.txt

# Cài riêng numpy để tránh conflict
pip install numpy==1.26.4
```

> **Lưu ý:** `requirements.txt` tải PyTorch từ index `nightly/cu128`.  
> Nếu Pod của bạn dùng CUDA version khác, hãy chỉnh dòng `--extra-index-url` tương ứng.

---

## 3. Cấu hình biến môi trường

```bash
cp .env.example .env
nano .env
```

Điền vào 2 biến bắt buộc:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx   # Token HuggingFace (cần để tải IP-Adapter)
STORAGE=/workspace/data              # Thư mục cache models (trùng mount point Network Volume)
```

Lấy token tại: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 4. Tải weights

`download_weights.py` tự động tải toàn bộ:

| Nguồn | Model | Đích lưu |
|---|---|---|
| `paingoat/swiftedit-pretrain` | SwiftEdit weights (inverse_ckpt-120k, sbv2_0.5, ip_adapter) | `./swiftedit_weights/` |
| `stabilityai/sd-turbo` | SD-Turbo (InverseModel base) | `$STORAGE/` |
| `Manojb/stable-diffusion-2-1-base` | SD 2.1 Base (AuxiliaryModel) | `$STORAGE/` |
| `h94/IP-Adapter` | IP-Adapter image encoder | `$STORAGE/` |

```bash
python download_weights.py
```

> Mạng RunPod rất nhanh (~1–2 Gbps), toàn bộ download thường hoàn thành trong **3–10 phút**.

---

## 5. Khởi chạy Gradio Web UI

```bash
python app.py
```

Trên bảng điều khiển RunPod → **Connect → HTTP Port 7860** để mở giao diện.

### Giao diện 3 cột sau khi bấm Edit:

| Cột | Nội dung |
|---|---|
| **Source Image** | Upload ảnh gốc + nhập Source Prompt |
| **Edited Image** | Ảnh sau khi chỉnh sửa |
| **Predicted Edit Mask 🔴** | Vùng mô hình dự đoán sẽ thay đổi (đỏ = edit region, tối = background) |

> Mask được tự động tính từ **hiệu số inverted noise** giữa source prompt và edit prompt,
> sau đó overlay lên ảnh gốc để bạn kiểm tra xem mô hình đang "nhìn vào đâu".

---

## 6. Chạy CLI (không cần Gradio)

Chỉnh thông số trong `infer.py` (dòng 110–113):

```python
img_path = "./assets/imgs_demo/woman_face.jpg"
src_p    = "woman"
edit_p   = "Taylor Swift"
scale_ta = 1
```

Rồi chạy:

```bash
python infer.py
```

Kết quả lưu tại `results/{src_prompt}/{edit_prompt}_SY_{strength}.png`.

---

## 7. Cấu trúc thư mục sau khi cài xong

```text
/workspace/SwiftEdit/
├── swiftedit_weights/
│   ├── inverse_ckpt-120k/unet_ema/    ← Inversion Network
│   ├── sbv2_0.5/                      ← SwiftBrushV2 UNet
│   └── ip_adapter_ckpt-90k/
│       └── ip_adapter.bin             ← IP-Adapter weights
├── .env                               ← HF_TOKEN + STORAGE
├── results/                           ← Kết quả tự động lưu
├── app.py                             ← Gradio Web UI (có mask visualization)
├── infer.py                           ← Logic inference
├── models.py                          ← InverseModel · AuxiliaryModel · IPSBV2Model
└── src/
    ├── mask_ip_controller.py          ← MaskController (attention rescaling)
    ├── mask_attention_processor.py    ← IP + mask-guided attention processor
    └── attention_processor.py         ← Base IP-Adapter attention
```

---

## 8. Luồng hoạt động của mô hình

```
Ảnh gốc + src_prompt + edit_prompt
        │
        ▼
  InverseModel (SD-Turbo UNet)
  → Encode ảnh → latent
  → Dự đoán inverted noise với cả 2 prompts song song
  → Tính |noise_src - noise_edit|.mean(dim=channel)
  → Clamp + normalize → binary mask12 (64×64)  ← MASK VISUALIZATION
        │
        ▼
  MaskController
  → Foreground (mask=1): scale_ip_fg = 0.2   ← vùng được edit
  → Background (mask=0): scale_ip_bg = 1.0   ← vùng giữ nguyên
        │
        ▼
  IPSBV2Model (SwiftBrushV2 + IP-Adapter)
  → 1 denoising step duy nhất
        │
        ▼
  Ảnh kết quả (512×512) — ~0.23s/ảnh trên A100
```

---

## 9. Tham số điều chỉnh

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `Edit Strength` (UI) / `scale_ta` | `1.0` | Cường độ chỉnh sửa tổng thể |
| `scale_edit` | `0.2` | Tỉ lệ IP-Adapter cho vùng edit (foreground) |
| `scale_non_edit` | `1.0` | Tỉ lệ IP-Adapter cho background |
| `mask_threshold` | `0.5` | Ngưỡng nhị phân hóa mask |
| `clamp_rate` | `3.0` | Kiểm soát tương phản của mask |

---

## 10. Xử lý sự cố thường gặp

### `CUDA out of memory`
- Đảm bảo VRAM ≥ 24 GB. Ảnh input luôn được resize về 512×512 nên không thể giảm thêm.
- Restart Pod và thử lại.

### `FileNotFoundError: swiftedit_weights/...`
- Chưa chạy `python download_weights.py` hoặc download bị lỗi giữa chừng.
- Kiểm tra `swiftedit_weights/` đã có đủ 3 thư mục con: `inverse_ckpt-120k/`, `sbv2_0.5/`, `ip_adapter_ckpt-90k/`.

### HuggingFace 401 / 403 khi tải weights
- Kiểm tra `HF_TOKEN` trong `.env` có đúng không.
- Token phải có quyền **read**.

### Port 7860 không truy cập được
- RunPod → Pod → **Connect** → kiểm tra **HTTP Port 7860** đã được expose.
- Nếu chưa, xoá Pod và tạo lại với port 7860 được khai báo ngay từ đầu.
