# GNN phát hiện đường dây rửa tiền trong mạng lưới giao dịch

Project mới được dựng lại theo mẫu `AntiMoneyLaunderingDetectionWithGNN`, nhưng mở rộng đầy đủ cho đồ án tốt nghiệp ngành AI.

## Mục tiêu

Hệ thống thực hiện:

1. Chuyển CSV hàng triệu giao dịch thành **directed graph**.
2. Node là tài khoản, edge là dòng tiền.
3. Tạo đặc trưng đa chiều cho từng tài khoản.
4. Train **GraphSAGE hoặc GAT** bằng **NeighborLoader subgraph sampling**.
5. Mô hình phân loại **tài khoản trung chuyển nghi ngờ**.
6. Từ node nghi ngờ, trích xuất suspicious subgraph và trực quan hóa thành mạng nhện.
7. Xuất model `.pth`, metrics `.json`, predictions `.csv`, và `web_graph.json` cho web.

> Lưu ý học thuật: GNN không trực tiếp predict “đường dây”. GNN predict node nghi ngờ; đường dây rửa tiền được suy diễn từ suspicious subgraph sau khi phân loại node.

## Cấu trúc

```text
aml_gnn_datn_new/
├── data/
│   ├── raw/              # đặt HI-Small_Trans.csv ở đây
│   ├── processed/
│   ├── models/           # model .pth và metrics
│   └── outputs/          # kết quả predict cho web
├── source/
│   ├── data_loader.py
│   ├── train.py
│   ├── sampled_gnn_train.py
│   ├── predict.py
│   ├── app.py
│   ├── metrics.py
│   ├── losses.py
│   ├── mlp_baseline.py
│   ├── network_utils.py
│   ├── visualize_detected_network.py
│   └── models/
│       └── gnn_models.py
├── scripts/
├── web/
├── requirements.txt
└── README.md
```

## Cài đặt

Khuyến nghị môi trường Python 3.10.

```bash
pip install -r requirements.txt
```

Nếu dùng CUDA 11.8, cài PyTorch trước:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install -r requirements.txt
```

Nếu máy bị lỗi NumPy với Torch trên Windows, dùng:

```bash
pip uninstall numpy -y
pip install numpy==1.26.4
```

## Dữ liệu

Đặt file:

```text
data/raw/HI-Small_Trans.csv
```

CSV cần có cột:

```text
Timestamp, From Bank, Account, To Bank, Account.1,
Amount Received, Receiving Currency, Amount Paid,
Payment Currency, Payment Format, Is Laundering
```

## Train GraphSAGE

```bash
python source/train.py --data-path data/raw/HI-Small_Trans.csv --model-type graphsage --epochs 30
```

Hoặc dùng entrypoint tương thích:

```bash
python source/sampled_gnn_train.py --data-path data/raw/HI-Small_Trans.csv --model-type graphsage --epochs 30
```

## Train GAT

```bash
python source/train.py --data-path data/raw/HI-Small_Trans.csv --model-type gat --epochs 30 --heads 4
```

## Test nhanh trên ít dòng

```bash
python source/train.py --data-path data/raw/HI-Small_Trans.csv --model-type graphsage --sample-rows 50000 --epochs 5
```

## Output sau train

```text
data/models/gnn_graphsage_model.pth
data/models/gnn_graphsage_bundle.pth
data/models/gnn_graphsage_metrics.json
```

Với GAT:

```text
data/models/gnn_gat_model.pth
data/models/gnn_gat_bundle.pth
data/models/gnn_gat_metrics.json
```

## Predict CSV để web dùng

```bash
python source/predict.py \
  --bundle-path data/models/gnn_graphsage_bundle.pth \
  --csv-path data/raw/HI-Small_Trans.csv \
  --output-dir data/outputs
```

Kết quả:

```text
data/outputs/account_predictions.csv
data/outputs/web_graph.json
data/outputs/inference_summary.json
```

## Chạy API cho frontend

```bash
uvicorn source.app:app --reload
```

API:

```text
POST /predict
```

Upload CSV, backend sẽ trả về summary và tạo `web_graph.json`.

## Metrics

Báo cáo gồm:

- Accuracy
- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- Confusion Matrix
- Specificity
- Balanced Accuracy
- Negative Predictive Value

Threshold được chọn bằng validation F1.

## Pipeline đồ án

```text
CSV giao dịch
↓
Build directed transaction graph
↓
Extract account-level node features
↓
GraphSAGE/GAT + NeighborLoader
↓
Predict suspicious intermediary accounts
↓
Extract suspicious subgraph
↓
Find connected suspicious structures
↓
Export spider-web graph for visualization
```
