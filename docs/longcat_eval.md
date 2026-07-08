## 评测结果


| Model                    | n-shot              | Score |
| ------------------------ | --------------- | --- |
| LongCat-Flash-Chat-Expertx2  | 5|85.88 |
| LongCat-Flash-Chat-Expertx2-Depth32      | 5|85.79 |

##  MMLU

### LongCat-Flash-Chat-Expertx2

| Groups          | Version | Filter | n-shot | Metric | Value      | Stderr       |
|-----------------|---------|--------|--------|--------|------------|--------------|
| mmlu            | 2       | none   |        | acc    | 0.8588     | ± 0.0028     |
| - humanities    | 2       | none   | 5      | acc ↑  | 0.8051     | ± 0.0056     |
| - other         | 2       | none   | 5      | acc ↑  | 0.8812     | ± 0.0056     |
| - social sciences | 2       | none   | 5      | acc ↑  | 0.9188     | ± 0.0049     |
| - stem          | 2       | none   | 5      | acc ↑  | 0.8582     | ± 0.0061     |


### LongCat-Flash-Chat-Expertx2-Depth32

| Groups          | Version | Filter | n-shot | Metric | Value      | Stderr       |
|-----------------|---------|--------|--------|--------|------------|--------------|
| mmlu            | 2       | none   |        | acc    | 0.8579     | ± 0.0028     |
| - humanities    | 2       | none   | 5      | acc ↑  | 0.8047     | ± 0.0056     |
| - other         | 2       | none   | 5      | acc ↑  | 0.8812     | ± 0.0055     |
| - social sciences | 2       | none   | 5      | acc ↑  | 0.9171     | ± 0.0049     |
| - stem          | 2       | none   | 5      | acc ↑  | 0.8563     | ± 0.0061     |

### C-eval

| Groups       | Version | Filter | n-shot | Metric    | Value      | Stderr     |
|--------------|---------|--------|--------|-----------|------------|------------|
| ceval-valid  | 2       | none   | 5      | acc ↑     | 0.8663     | ± 0.009    |
|              |         | none   | 5      | acc_norm ↑| 0.8663     | ± 0.009    |