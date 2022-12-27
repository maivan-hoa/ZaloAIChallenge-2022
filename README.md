# Guide
Team mình cảm thấy rất may mắn khi đạt được top 2 Final; cũng như đồng top 1 trong LB Public test 1 và top 30 LB Public test 2. Sau đây là solution của team:

1. Xử lý dữ liệu: Team mình chia tập train-val đơn giản theo tỷ lệ 80/20 theo id video và cắt lấy 1 frame trên 1s.
2. Huấn luyện
- Public test 1: Giai đoạn đầu tụi mình xây dựng mô hình baseline với mô hình Efficientnet-B4 Noisy Student input size 512, sử dụng một số augmentation đơn giản như H/V Flip, ColorJitter. Mô hình đã fit được tập public test 1 rất tốt. Với một vài lần submit và thử nghiệm tay (với một chút may mắn) metric EER, team mình đã có được ground-truth của tập này.
- Public test 2: Với tập dữ liệu mới này, team mình đánh giá dữ liệu có chất lượng video kém hơn so với trước, cảm giác như video bị cắt nhỏ, làm mờ, … Nên tụi mình đưa ra 1 số hướng để giải quyết.
  - Phương pháp 1 (PP1): Bổ sung vào tập Val hiện tại các augs của chính nó, cố định thành một tập Val-aug mới (số lượng x2 Val cũ).
  - Phương pháp 2 (PP2): Các aug cho tập Val được tùy biến mỗi khi load dữ liệu (tương tự cách aug cho tập Train).
Mục tiêu của 2 hướng trên là tìm ra một không gian Augmentation của tập Val miêu tả chính xác nhất, team mình sử dụng mô hình Swin Transformer để huấn luyện cho cả 2 phương pháp trên.

Mô hình Swin_PP1:
  - Kiến trúc: swin_large_patch4_window12_384 với drop_path_rate: 0.3
  - Optimizer: LR: 1e-5, Weight decay: 3e-5
  - LR Scheduler: CosineAnnealingWarmRestarts với T0: 600, T_mult: 1, Eta_min: 1e-7
  - Train Augmentation: RandomResizedCrop 0.49-1.0, RandomHorizontalFlip 0.5, RandomVerticalFlip 0.2, GaussianBlur kernel_size 3 sigma 0.2-2.0, ColorJitter contrast 0.2.
  - Offline-Augmentation Val: Downsize Image: CenterCrop 0.75 * height - 0.75 * width, Resize 0.375 * height - 0.375 * width.
  - Chuẩn bị dữ liệu Val-aug = tập val gốc + offline-augmentation val.
  - Trainer: Mixed Precision với FP16, Batch size 16, Max 12000 step, validation mỗi epoch, lưu 2 mô hình val loss thấp nhất, lấy mô hình với val accuracy cao nhất trong 2.

Mô hình Swin_PP2:
  - Kiến trúc: swin_large_patch4_window12_384
  - Optimizer: LR: 3e-5, Weight decay: 1e-6
  - LR Scheduler: StepLR với step_size: 5, gamma: 0.2
  - Train / Val Augmentation: RandomResizedCrop 0.49-1.0, RandomVerticalFlip 0.2, GaussianBlur kernel_size 3 sigma 0.2-2.0.
  - Trainer: Mixed Precision với FP16, Batch size 16, Max 20 epoch, validation mỗi epoch, lưu 3 mô hình val loss thấp nhất và last checkpoint. Qua thử nghiệm tụi mình chọn last checkpoint.
  
Tụi mình đánh giá 2 mô hình Swin_PP1 và Swin_PP2 nhận thấy mô hình PP1 tốt trên public test 1, mô hình PP2 lại tốt trên public test 2 nên team đã ensemble 2 mô hình lại và thấy mô hình Ensemble này cho kết quả (EER và TEER) tốt hơn.

Một số điểm tốt mà tụi mình đánh giá đem lại thành công cho mô hình:
Training với Mixed Precision. Team mình từ đầu cuộc thi cũng cân nhắc về time constraint và chú ý training với mixed precision, nó giúp các mô hình tụi mình inference nhanh nhưng hiệu quả không kém mô hình FP32.
Ensemble: tụi mình chỉ muốn ensemble 2 mô hình tốt nhất trên các LB, nhưng có lẽ nên dùng nhiều hơn.
Augmentation offline và online: team xác định các loại augs phù hợp và tuning 🙂
May mắn: team khá gặp may mắn khi hoàn thành được codebase và baseline khá sớm, cũng như một số may mắn ở trên, tụi mình chỉ phân tích lỗi và để nó tuning liên tục thôi :’)


# Cài đặt

## Cài đặt pytorch
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Cài đặt các library

```bash
pip install -r pip_env.txt
```

## Cài đặt dữ liệu

### Bước 1:
- Tải file train.zip để vào trong thư mục data/
- Tiền xử lý dữ liệu: Cắt các frame với 1 frame/s.
- Chia dữ liệu

```bash
cd data/
unzip train.zip
python get_frame.py -i train/videos/ -o train/images/
python create_data_h.py -dir train -images images -l label.csv
python create_data_s.py -dir train -images images -l label.csv
```

### Bước 2: Tạo data augmentation cho tập train.

```bash
cd ..
python val_augmentation.py
```

**Note**: ta có thể sử dụng bash file. 
```bash
bash process_data.sh
```

# Hướng tiếp cận

Team sử dụng 2 hướng tiếp cận chính:

- Augmentation tập validation offline và giữ nguyên nó trong quá trình huấn luyện mô hình (mô hình h). Mục tiêu chính là team nhận thấy tập validation được sinh ra gần với tập public test 1 và 2 nhất và cho ra kết quả tốt nhất.

- Augmentation cả tập train và validation online trong lúc huấn luyện (mô hình s). Mục tiêu chính là để lấy được mô hình tốt nhất trên nhiều không gian khác nhau.

# Huấn luyện mô hình

## Mô hình h

```bash
CUDA_VISIBLE_DEVICES=1 python main_h.py
```

- Output sẽ là ở **outputs/h/ckpts**.
- Ở đây team mình lựa chọn mô hình cho ra **val_acc** lớn nhất (được đánh giá là tốt nhất trên tập public 1 và 2).

## Mô hình s

```bash
CUDA_VISIBLE_DEVICES=1 python main_s.py
```

- Output sẽ là ở **outputs/s/ckpts**.
- Ở đây team mình lựa chọn mô hình last.ckpt.

**Note**: Kết quả huấn luyện trên các device khác nhau có thể sai khác. Tụi mình đã có set seed và để CUDA Benchmark để mỗi lần huấn luyện là giống nhau trên 1 device.

# Inference

```bash
cp -r outputs/h weights/
cp -r outputs/s weights/
```

Sửa lại file config ở trong configs/inference.yaml.

```yaml
checkpoint_s: <Path to s folder>
checkpoint_h: <Path to h folder>
videos_dir: private_test/videos/*

hydra:
  run:
    dir: /result/
```

Chạy ensemble  cả 2 mô hình để predict.

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py
```
