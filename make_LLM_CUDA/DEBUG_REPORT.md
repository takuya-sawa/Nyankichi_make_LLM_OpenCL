# CUDA版TinyLLM デバッグレポート

**日付**: 2025年12月14日  
**ビルド**: Dec 14 2025 14:33:48  
**ステータス**: 🔴 未解決

---

## 問題の概要

CUDA実装のTransformerモデルが訓練・推論時に常に`<pad>`トークンのみを出力する。
原因は**最終的なLogitsが全てゼロ**になるため。

### 症状
- 訓練Loss: 変動するが改善なし (-0.99 ~ 0.7程度)
- 推論結果: 全て `<pad>` トークン
- Logits: `Max: 0, Min: 0` (完全にゼロ)

---

## 根本原因の特定

### ✅ 正常に動作している箇所

1. **TransformerLayer::Forward()の出力**
   ```
   [DEBUG] ff_out.d_data[0:5]: -0.000100224 3.2093e-05 0.000404273
   ```
   → レイヤーの計算自体は**正しく実行されている**

2. **Tensor代入操作のログ**
   ```
   [TRACE] Assignment operator: used h_data to initialize GPU (2048 bytes)
   ```
   → 代入演算子は正常に呼び出され、h_dataからGPUへの転送を報告

3. **重みの初期化**
   - 範囲: ±0.1（Xavier風）
   - 確認済み: アンダーフローなし

### ❌ 問題が発生している箇所

**TinyLLM::Forward()のレイヤーループ後**

```cpp
// 355-370行目付近
Tensor x = embedded;  // 初期値は正しい
for (int layer_idx = 0; layer_idx < 3; layer_idx++) {
    Tensor layer_out = layers[layer_idx]->Forward(x);  // ✅ 出力は正しい
    x = layer_out;  // ⚠️ 代入演算子実行
    x.h2d();  // ⚠️ CPU→GPU転送
}
// ループ終了後...
// [DEBUG] x.d_data[0:5]: 6.16847e-33 1.60859e-32  ❌ ゼロ！
```

**矛盾点**:
- `ff_out.d_data`には正しい値がある（-0.0001, 3.2e-05, 0.0004...）
- 代入演算子は「h_dataからGPUへ転送した」と報告
- **しかし**、`x.d_data`は完全にゼロ（6.16e-33 ≈ ゴミ値/未初期化）

---

## これまでの修正履歴

### 1️⃣ Tensorコピーコンストラクタ修正 ✅
**問題**: GPU→GPU コピーでなくCPU→GPU コピーをしていた  
**修正**: `cudaMemcpyDeviceToDevice` を使用  
**結果**: コピー自体は正しく動作

### 2️⃣ TransposeMatrix CPU実装 ✅
**問題**: `kernel_transpose`がデータを書き込まない  
**修正**: CPU側で転置処理を実装  
**結果**: 転置は正常動作（K_T確認済み）

### 3️⃣ 重み初期化スケール変更 ✅
**問題**: ±0.005では小さすぎてアンダーフロー  
**修正**: ±0.1に変更  
**結果**: 重みは適切な範囲

### 4️⃣ デバッグ用h2d()削除 ✅
**問題**: デバッグ出力後にh2d()を呼び、ゼロでGPUを上書き  
**修正**: 不要なh2d()呼び出しを削除  
**結果**: デバッグによる破壊は解消

### 5️⃣ 戻り値ライフタイム対策 ⚠️ 部分的
**問題**: `return ff_out;` でデストラクタがGPUメモリを解放  
**修正案A**: `ff_out.d2h()` before return + `x.h2d()` after assignment  
**結果**: ログは成功を報告するが、実際のGPUデータはゼロ

**修正案B**: コピーコンストラクタ/代入演算子でh_data優先  
**結果**: 同じくゼロ

**修正案C**: d2h()で自動リサイズ  
**結果**: 同じくゼロ

---

## 現在の疑問点

### 🤔 なぜh_dataからの転送が失敗するのか？

1. **可能性1: h_dataが空またはゼロ**
   - 代入演算子は「h_dataからGPUへ転送」と報告
   - しかし`h_data`の中身は確認していない
   - **次のデバッグ**: `h_data[0:5]`を出力して内容確認

2. **可能性2: h2d()が実際には何もしていない**
   ```cpp
   void h2d() {
       if (!h_data.empty() && d_data != nullptr) {
           cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);
       }
   }
   ```
   - cudaMemcpyのエラーチェックなし
   - **次のデバッグ**: cudaMemcpyの戻り値を確認

3. **可能性3: d2h()がh_dataを正しく埋めていない**
   ```cpp
   void d2h() {
       if (d_data != nullptr && size > 0) {
           if (h_data.size() != size) h_data.resize(size);
           cudaMemcpy(h_data.data(), d_data, size, cudaMemcpyDeviceToHost);
       }
   }
   ```
   - sizeの単位が間違っている可能性（要素数 vs バイト数）
   - **次のデバッグ**: d2h()直後にh_data[0:5]を出力

4. **可能性4: メモリアライメントまたはサイズ計算エラー**
   - `size`は`sizeof(float) * rows * cols`で計算
   - しかし`h_data.size()`は要素数（float数）
   - **ミスマッチ**: `h_data.resize(size)` → バイト数でリサイズ？
   - **次の確認**: sizeの意味を全コードで統一確認

### 🚨 最も疑わしいバグ

**tensor_cuda.cu の d2h() 実装**:
```cpp
void Tensor::d2h() {
    if (d_data != nullptr && size > 0) {
        if (h_data.size() != size) h_data.resize(size);  // ⚠️ ここ！
        cudaMemcpy(h_data.data(), d_data, size, cudaMemcpyDeviceToHost);
    }
}
```

**問題点**:
- `size` = `sizeof(float) * rows * cols` （バイト数）
- `h_data` = `std::vector<float>` （要素数でカウント）
- `h_data.resize(size)` → **バイト数で要素数をリサイズ！**
- 例: 512要素 = 2048バイト → `h_data.resize(2048)` = 2048要素になる（**4倍のサイズ**）

**期待する修正**:
```cpp
void Tensor::d2h() {
    if (d_data != nullptr && size > 0) {
        size_t num_elements = size / sizeof(float);  // バイト→要素数変換
        if (h_data.size() != num_elements) {
            h_data.resize(num_elements);
        }
        cudaMemcpy(h_data.data(), d_data, size, cudaMemcpyDeviceToHost);
    }
}
```

---

## 次のアクションプラン

### 優先度1: サイズ計算バグの確認と修正 🔥
1. `Tensor::d2h()`のresizeを修正（バイト数→要素数変換）
2. `Tensor::h2d()`も同様にチェック
3. 全cudaMemcpyでエラーチェック追加

### 優先度2: デバッグ出力の強化
1. `d2h()`呼び出し直後に`h_data[0:5]`と`h_data.size()`を出力
2. `h2d()`呼び出し前に`h_data[0:5]`と`h_data.size()`を出力
3. cudaMemcpyの戻り値を確認して失敗時はエラーメッセージ

### 優先度3: 代替アプローチ
もしサイズバグが原因でなければ:
- 戻り値を参照渡しに変更（`void Forward(Tensor& out)`）
- または`std::shared_ptr<Tensor>`で管理してメモリ共有

---

## コード断片（参考用）

### TransformerLayer::Forward() 終了部（280行目付近）
```cpp
ff_out.d2h();  // GPU→CPU転送
return ff_out;  // コピーコンストラクタ呼び出し
```

### TinyLLM::Forward() ループ部（355-370行目）
```cpp
Tensor x = embedded;
for (int layer_idx = 0; layer_idx < 3; layer_idx++) {
    Tensor layer_out = layers[layer_idx]->Forward(x);
    x = layer_out;  // 代入演算子
    x.h2d();  // CPU→GPU復元
}
// この後xはゼロ！
```

### Tensor コピーコンストラクタ（28-55行目付近）
```cpp
Tensor::Tensor(const Tensor& other) {
    h_data = other.h_data;  // CPU側コピー
    allocate();
    if (!h_data.empty() && d_data != nullptr) {
        // h_dataからGPUへ転送（優先）
        cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);
    } else if (other.d_data != nullptr && d_data != nullptr) {
        // GPU→GPU直接コピー（フォールバック）
        cudaMemcpy(d_data, other.d_data, size, cudaMemcpyDeviceToDevice);
    }
}
```

---

## 環境情報

- **OS**: Windows
- **CUDA**: 13.1
- **コンパイラ**: NVCC + MSVC 2022
- **GPU**: （不明）
- **cuBLAS**: 初期化成功（現在はCPU fallbackで無効化中）

---

## メモ

- cuBLASを無効化してもゼロ問題は解決しない → 行列計算の問題ではない
- matmulはCPUで正しく動作（K, Q, Vの値確認済み）
- AttentionとFFN内部の計算は正常（ff_out確認済み）
- **問題はデータ転送（GPU↔CPU）またはオブジェクトライフタイム管理**

---

## 結論

**最有力仮説**:  
`Tensor::d2h()`で`h_data.resize(size)`を呼び出す際、`size`（バイト数）を要素数と誤認してリサイズしている。これにより4倍のサイズにリサイズされ、実際のデータは先頭1/4のみに格納され、残りはゼロまたは未初期化。その結果、`h2d()`で転送されるデータの大部分がゼロになる。

**即座に試すべき修正**:
```cpp
// tensor_cuda.cu の d2h() と h2d() を修正
size_t num_elements = size / sizeof(float);
h_data.resize(num_elements);  // 要素数でリサイズ
```

この修正で問題が解決する可能性は**90%以上**。
