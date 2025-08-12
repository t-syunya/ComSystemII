#!/bin/bash

echo "=== ループアンローリング実装の修正 ==="
echo "重複関数を削除して、正しい実装を追加します"
echo ""

# バックアップ作成
echo "バックアップを作成中..."
cp gemm-test.c gemm-test.c.backup.$(date +%Y%m%d_%H%M%S)
echo "✓ バックアップ作成完了"
echo ""

# 既存の重複関数を削除
echo "既存の重複関数を削除中..."

# dgemm_unroll関数の削除（最初の実装）
if grep -n "void dgemm_unroll" gemm-test.c | wc -l | grep -q "2"; then
    echo "重複するdgemm_unroll関数を削除中..."
    # 最初のdgemm_unroll関数を削除
    sed -i '/^void dgemm_unroll(REAL \*A, REAL \*B, REAL \*C, int n)$/,/^}$/d' gemm-test.c
    echo "✓ 重複関数削除完了"
else
    echo "重複関数は見つかりませんでした"
fi

# 他の重複関数も削除
if grep -n "void dgemm_blocking_unroll" gemm-test.c | wc -l | grep -q "2"; then
    sed -i '/^void dgemm_blocking_unroll(REAL \*A, REAL \*B, REAL \*C, int n)$/,/^}$/d' gemm-test.c
fi

if grep -n "void dgemm_OMP_unroll" gemm-test.c | wc -l | grep -q "2"; then
    sed -i '/^void dgemm_OMP_unroll(REAL \*A, REAL \*B, REAL \*C, int n)$/,/^}$/d' gemm-test.c
fi

echo ""

# 正しいループアンローリング実装を追加
echo "正しいループアンローリング実装を追加中..."

# AVX_OMP関数の後に挿入
insert_point=$(grep -n "dgemm_AVX_OMP" gemm-test.c | tail -1 | cut -d: -f1)
if [ -z "$insert_point" ]; then
    echo "❌ AVX_OMP関数が見つかりません"
    exit 1
fi

# 関数の終了位置を特定
end_line=$(sed -n "$insert_point,\$p" gemm-test.c | grep -n "^}" | head -1 | cut -d: -f1)
actual_insert_point=$((insert_point + end_line))

# 正しい実装を挿入
cat > temp_correct_unroll.c << 'EOF'
/* Loop Unrolling - 正しい実装（列優先アクセス） */
void dgemm_unroll(REAL *A, REAL *B, REAL *C, int n)
{
  int i, j, k;
  int unroll_factor = 4;  // 固定値ではなく変数として定義
  
  // 初期化
  for (i = 0; i < n * n; i++)
    C[i] = 0.0;
  
  // 列優先アクセスパターン（正しい実装）
  for (j = 0; j < n; j++)
    for (k = 0; k < n; k++)
    {
      REAL bkj = B[k + j * n];  // B[k][j]を一度読み込み
      // アンロール可能な部分
      for (i = 0; i <= n - unroll_factor; i += unroll_factor)
      {
        C[i + j * n] += A[i + k * n] * bkj;
        C[(i+1) + j * n] += A[(i+1) + k * n] * bkj;
        C[(i+2) + j * n] += A[(i+2) + k * n] * bkj;
        C[(i+3) + j * n] += A[(i+3) + k * n] * bkj;
      }
      // 残りの要素を処理
      for (; i < n; i++)
        C[i + j * n] += A[i + k * n] * bkj;
    }
}

/* より効率的なループアンローリング（最適化版） */
void dgemm_unroll_optimized(REAL *A, REAL *B, REAL *C, int n)
{
  int i, j, k;
  
  // 初期化
  for (i = 0; i < n * n; i++)
    C[i] = 0.0;
  
  // 列優先アクセスパターン（最適化版）
  for (j = 0; j < n; j++)
    for (k = 0; k < n; k++)
    {
      REAL bkj = B[k + j * n];  // B[k][j]を一度読み込み
      // より大きなアンロール係数（8）
      for (i = 0; i <= n - 8; i += 8)
      {
        C[i + j * n] += A[i + k * n] * bkj;
        C[(i+1) + j * n] += A[(i+1) + k * n] * bkj;
        C[(i+2) + j * n] += A[(i+2) + k * n] * bkj;
        C[(i+3) + j * n] += A[(i+3) + k * n] * bkj;
        C[(i+4) + j * n] += A[(i+4) + k * n] * bkj;
        C[(i+5) + j * n] += A[(i+5) + k * n] * bkj;
        C[(i+6) + j * n] += A[(i+6) + k * n] * bkj;
        C[(i+7) + j * n] += A[(i+7) + k * n] * bkj;
      }
      // 残りの要素を処理
      for (; i < n; i++)
        C[i + j * n] += A[i + k * n] * bkj;
    }
}

/* Blocking + Loop Unrolling */
void do_block_unroll(int n, int si, int sj, int sk, REAL *A, REAL *B, REAL *C)
{
  int i, j, k;
  int unroll_factor = 4;

  for (i = si; i < si + BLOCKSIZE; ++i)
  {
    for (j = sj; j < sj + BLOCKSIZE; ++j)
    {
      for (k = sk; k < sk + BLOCKSIZE - unroll_factor + 1; k += unroll_factor)
      {
        C[i + j * n] += A[i + k * n] * B[k + j * n];
        C[i + j * n] += A[i + (k+1) * n] * B[(k+1) + j * n];
        C[i + j * n] += A[i + (k+2) * n] * B[(k+2) + j * n];
        C[i + j * n] += A[i + (k+3) * n] * B[(k+3) + j * n];
      }
      // 残りの要素を処理
      for (; k < sk + BLOCKSIZE; k++)
        C[i + j * n] += A[i + k * n] * B[k + j * n];
    }
  }
}

void dgemm_blocking_unroll(REAL *A, REAL *B, REAL *C, int n)
{
  int sj, si, sk;

  for (sj = 0; sj < n; sj += BLOCKSIZE)
    for (si = 0; si < n; si += BLOCKSIZE)
      for (sk = 0; sk < n; sk += BLOCKSIZE)
        do_block_unroll(n, si, sj, sk, A, B, C);
}

/* OpenMP + Loop Unrolling */
void dgemm_OMP_unroll(REAL *A, REAL *B, REAL *C, int n)
{
  int i, j, k;
  int unroll_factor = 4;

  // 初期化
  for (i = 0; i < n * n; i++)
    C[i] = 0.0;

#pragma omp parallel for private(j,k)
  for (i = 0; i < n; i++)
  {
    for (j = 0; j < n; j++)
    {
      for (k = 0; k <= n - unroll_factor; k += unroll_factor)
      {
        C[i + j * n] += A[i + k * n] * B[k + j * n];
        C[i + j * n] += A[i + (k+1) * n] * B[(k+1) + j * n];
        C[i + j * n] += A[i + (k+2) * n] * B[(k+2) + j * n];
        C[i + j * n] += A[i + (k+3) * n] * B[(k+3) + j * n];
      }
      // 残りの要素を処理
      for (; k < n; k++)
        C[i + j * n] += A[i + k * n] * B[k + j * n];
    }
  }
}

EOF

# ファイルに挿入
sed -i "${actual_insert_point}a\\" gemm-test.c
sed -i "${actual_insert_point}r temp_correct_unroll.c" gemm-test.c
rm temp_correct_unroll.c

echo "✓ 正しい実装追加完了"
echo ""

# コンパイルテスト
echo "コンパイルテスト中..."
make clean
make
if [ $? -eq 0 ]; then
    echo "✓ コンパイル成功"
else
    echo "❌ コンパイルエラー"
    exit 1
fi

echo ""
echo "=== 修正完了 ==="
echo "正しいループアンローリング実装が追加されました"
echo ""
echo "テストを実行しますか？ (y/n): "
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo ""
    echo "=== テスト実行 ==="
    
    # 修正版ループアンローリングのテスト
    echo "修正版ループアンローリングを有効化..."
    sed -i 's|//#define UNROLL_ONLY|#define UNROLL_ONLY|' gemm-test.c
    make clean && make
    make run
    
    echo ""
    echo "最適化版ループアンローリングを有効化..."
    sed -i 's|#define UNROLL_ONLY|//#define UNROLL_ONLY|' gemm-test.c
    sed -i 's|//#define UNROLL_OPTIMIZED|#define UNROLL_OPTIMIZED|' gemm-test.c
    make clean && make
    make run
    
    # 元に戻す
    sed -i 's|#define UNROLL_OPTIMIZED|//#define UNROLL_OPTIMIZED|' gemm-test.c
    
    echo ""
    echo "=== テスト完了 ==="
fi
