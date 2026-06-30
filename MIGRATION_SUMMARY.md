# Migration Summary

Dự án này được tạo mới hoàn toàn dựa trên ý tưởng của repo mẫu `AntiMoneyLaunderingDetectionWithGNN`.

## Giữ ý tưởng từ repo mẫu

- Dùng PyTorch Geometric.
- Dùng GAT cho AML detection.
- Dùng NeighborLoader để training theo sampling.
- Dữ liệu đầu vào là `HI-Small_Trans.csv`.

## Nâng cấp cho đồ án tốt nghiệp

| Yêu cầu | File xử lý |
|---|---|
| Mô hình hóa CSV thành directed graph | `source/data_loader.py` |
| Node là tài khoản, edge là dòng tiền | `source/data_loader.py` |
| Feature đa chiều trên node | `source/data_loader.py` |
| GraphSAGE và GAT | `source/models/gnn_models.py` |
| Subgraph sampling | `source/train.py` |
| Xuất model bundle | `source/train.py` |
| Metrics mất cân bằng | `source/metrics.py` |
| Predict CSV upload | `source/predict.py` |
| Trích suspicious subgraph | `source/network_utils.py` |
| Xuất `web_graph.json` | `source/network_utils.py` |
| API web upload CSV | `source/app.py` |
| MLP baseline | `source/mlp_baseline.py` |

## Điểm quan trọng để bảo vệ

GNN thực hiện node classification để tìm tài khoản trung chuyển nghi ngờ. Sau đó hệ thống trích xuất subgraph từ các node nghi ngờ để trực quan hóa mạng lưới rửa tiền.
