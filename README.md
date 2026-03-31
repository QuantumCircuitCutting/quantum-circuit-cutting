# Quantum Circuit Cutting

大規模な量子回路を小さなサブ回路に分割し、古典シミュレータまたは量子デバイスで
実行した後、結果を再構築するライブラリです。

## なぜ回路分割が必要か？

現在の量子コンピュータにはqubit数や接続性に制約があり、
大きな量子回路をそのまま実行できないことがあります。
また古典シミュレータでも、qubit数が増えるとメモリが指数的に増大します。

回路分割（Circuit Cutting）は、大きな回路を小さなサブ回路に分割して
個別に実行し、結果を古典的に再構築する手法です。
これにより、現在のハードウェアの制約を超えた量子計算が可能になります。

## 特徴

- **2つの実行パス**: 古典シミュレータ（状態ベクトル）と量子デバイス（実機）に対応
- **自動カット位置探索**: メモリ制約やqubit数制約に基づいて最適なカット位置を自動決定
- **Gate Cut / Wire Cut**: 2種類のカット手法に対応
- **デバイス推薦**: サブ回路に最適な量子デバイスを自動選択
- **テンソルネットワーク再構築**: 量子パスの測定結果から元の回路の期待値を復元

## インストール

```
pip install quantum-circuit-cutting
```


| パッケージ | 機能 |
|---|---|
| qcc-preprocess | 前処理（2qubitゲート削減） |
| qcc-cut-locator | カット位置の自動探索 |
| qcc-circuit-cutter | サブ回路生成（gate cut / wire cut） |
| qcc-device-recommender | デバイス推薦 |
| qcc-circuit-transformer | 回路変換・最適化 |
| qcc-classical | 古典シミュレーションパス |
| qcc-treewidth-gate-cut | Treewidthベースのgate cut |
| qcc-annealing-transpiler | アニーリングベースのトランスパイラ |

## Quick Start

[quickstart.ipynb](https://github.com/QuantumCircuitCutting/qcc-tutorials/blob/main/quickstart.ipynb) を参照してください。

## パイプライン

```
機能1: 前処理（スキップ可）
  |
機能2: 回路分割（カット位置探索 + サブ回路生成）
  |
  +---> 古典パス ---> 機能5: 結果再構築
  |
  +---> 量子パス --->
          機能3: デバイス推薦（スキップ可）
          機能4: 量子実行
          機能5: 結果再構築
```

## ライセンス

CC0 1.0 Universal. See [LICENSE](LICENSE).
