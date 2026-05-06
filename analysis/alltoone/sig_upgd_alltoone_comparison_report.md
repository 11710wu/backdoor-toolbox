# SIG/UPGD all-to-one 对比分析报告

## 1. 数据覆盖与配对成功率
- 提取总行数: 630
- new 行数: 270
- baseline 行数: 360
- 配对键总数: 360
- 成功配对键数: 270
- 仅 new 存在键数: 0
- 仅 baseline 存在键数: 90
- 字段缺失行数: 0

## 2. 分组变化汇总（按 dataset/model/attack）
|dataset|model|attack|pair_count|transfer_delta_mean|asr_delta_mean|S_stealth_delta_mean|
|---|---|---:|---:|---:|---:|---:|
|cifar10|mobilenet|sig|21|0.4384598571428571|0.32336994300013905||
|cifar10|mobilenet|upgd|24|0.23796229166666666|0.1553593771722508||
|cifar10|resnet18|sig|21|0.34081090476190473|0.2687268707010122||
|cifar10|resnet18|upgd|24|0.15393866666666667|0.11527410908753881||
|cifar10|vgg|sig|21|0.3674775714285714|0.26801851005938315||
|cifar10|vgg|upgd|24|0.18600908333333335|0.17640993558552298||
|mnistm|mobilenet|sig|21|0.7635132380952381|0.7127702002073861||
|mnistm|mobilenet|upgd|24|0.354847125|0.5275160524846455||
|mnistm|resnet18|sig|21|0.8588842380952381|0.8458562654542554||
|mnistm|resnet18|upgd|24|0.66320025|0.5042399497487438||
|mnistm|vgg|sig|21|0.6391547142857142|0.7576905692483582||
|mnistm|vgg|upgd|24|0.7889562083333334|0.5821235808672994||

## 3. 关键变化 Top10（按 |transfer_rate_delta|）
|pair_id|dataset|model|attack|poison_rate|target_label|transfer_delta|asr_delta|
|---|---|---|---|---:|---:|---:|---:|
|mnistm|resnet18|upgd|0.01|||4.0|100.0|5.0#1|mnistm|resnet18|upgd|0.01|2|0.998465|0.8708821887213847|
|mnistm|vgg|upgd|0.005|||12.0|100.0|5.0#1|mnistm|vgg|upgd|0.005|2|0.9979060000000001|0.8235622557230597|
|mnistm|vgg|sig|0.05|6.0|56.0|||#1|mnistm|vgg|sig|0.05|2|0.997766|0.8058347292015634|
|mnistm|resnet18|sig|0.05|6.0|12.0|||#1|mnistm|resnet18|sig|0.05|2|0.997348|0.9811557788944724|
|mnistm|vgg|upgd|0.01|||12.0|100.0|5.0#1|mnistm|vgg|upgd|0.01|2|0.997208|0.8813512004466778|
|mnistm|mobilenet|sig|0.05|6.0|28.0|||#1|mnistm|mobilenet|sig|0.05|2|0.996929|0.9798994974874371|
|mnistm|vgg|upgd|0.05|||8.0|100.0|5.0#1|mnistm|vgg|upgd|0.05|2|0.99679|0.876465661641541|
|mnistm|vgg|sig|0.05|6.0|44.0|||#1|mnistm|vgg|sig|0.05|2|0.996231|0.8661362367392519|
|mnistm|resnet18|upgd|0.01|||8.0|100.0|5.0#1|mnistm|resnet18|upgd|0.01|2|0.995952|0.8192350642099386|
|mnistm|vgg|upgd|0.05|||4.0|100.0|5.0#1|mnistm|vgg|upgd|0.05|2|0.995952|0.8997766610831938|

## 4. 不可比与异常样本
- unmatched 行数: 90

|side|dataset|model|attack|poison_rate|target_label|folder_name|
|---|---|---|---|---:|---:|---|
|baseline_only|tiny_imagenet|mobilenet|sig|0.001||SIG_0.001_delta=4_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|mobilenet|sig|0.001||SIG_0.001_delta=12_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|mobilenet|sig|0.001||SIG_0.001_delta=20_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|mobilenet|sig|0.001||SIG_0.001_delta=28_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|mobilenet|sig|0.001||SIG_0.001_delta=36_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|mobilenet|sig|0.001||SIG_0.001_delta=44_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|mobilenet|sig|0.001||SIG_0.001_delta=56_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|mobilenet|sig|0.005||SIG_0.005_delta=4_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|mobilenet|sig|0.005||SIG_0.005_delta=12_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|mobilenet|sig|0.005||SIG_0.005_delta=20_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|mobilenet|sig|0.005||SIG_0.005_delta=28_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|mobilenet|sig|0.005||SIG_0.005_delta=36_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|mobilenet|sig|0.005||SIG_0.005_delta=44_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|mobilenet|sig|0.005||SIG_0.005_delta=56_f=6_poison_seed=2333_arch=mobilenetv2_tiny_imagenet|
|baseline_only|tiny_imagenet|resnet18|sig|0.001||SIG_0.001_delta=4_f=6_poison_seed=2333_arch=ResNet18_tiny_imagenet|
|baseline_only|tiny_imagenet|resnet18|sig|0.001||SIG_0.001_delta=12_f=6_poison_seed=2333_arch=ResNet18_tiny_imagenet|
|baseline_only|tiny_imagenet|resnet18|sig|0.001||SIG_0.001_delta=20_f=6_poison_seed=2333_arch=ResNet18_tiny_imagenet|
|baseline_only|tiny_imagenet|resnet18|sig|0.001||SIG_0.001_delta=28_f=6_poison_seed=2333_arch=ResNet18_tiny_imagenet|
|baseline_only|tiny_imagenet|resnet18|sig|0.001||SIG_0.001_delta=36_f=6_poison_seed=2333_arch=ResNet18_tiny_imagenet|
|baseline_only|tiny_imagenet|resnet18|sig|0.001||SIG_0.001_delta=44_f=6_poison_seed=2333_arch=ResNet18_tiny_imagenet|

## 5. 输出文件索引
- `analysis/alltoone/data_sig_upgd_alltoone_raw.csv`
- `analysis/alltoone/validation_sig_upgd_alltoone.json`
- `analysis/alltoone/report_tables/sig_upgd_alltoone_pairwise_comparison.csv`
- `analysis/alltoone/report_tables/sig_upgd_alltoone_group_summary.csv`
- `analysis/alltoone/report_tables/sig_upgd_alltoone_unmatched_cases.csv`
