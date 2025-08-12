#!/bin/bash

# ループアンローリング実装スクリプト
# gemm-test.cにループアンローリング機能を追加します

set -e

# 色付き出力
print_header() {
    echo -e "\n\033[1;34m=== $1 ===\033[0m"
}

print_success() {
    echo -e "\033[1;32m✓ $1\033[0m"
}

print_info() {
    echo -e "\033[1;36mℹ $1\033[0m"
}

print_warning() {
    echo -e "\033[1;33m⚠ $1\033[0m"
}

print_error() {
    echo -e "\033[1;31m✗ $1\033[0m"
}

# バックアップ作成
create_backup() {
    print_info "元ファイルのバックアップを作成中..."
    cp gemm-test.c gemm-test.c.backup.$(date +%Y%m%d_%H%M%S)
    print_success "バックアップ作成完了"
}

# ループアンローリング関数の追加
add_unroll_function() {
    print_info "ループアンローリング関数を追加中..."
    
    # AVX_OMP関数の後に追加
    local insert_point=$(grep -n "dgemm_AVX_OMP" gemm-test.c | tail -1 | cut -d: -f1)
    
    if [ -z "$insert_point" ]; then
        print_error "AVX_OMP関数が見つかりません"
        return 1
    fi
    
    # 挿入位置を計算（関数の終了後）
    local end_line=$(sed -n "$insert_point,\$p" gemm-test.c | grep -n "^}" | head -1 | cut -d: -f1)
    local actual_insert_point=$((insert_point + end_line))
    
    # ループアンローリング関数を挿入
    cat > temp_unroll_function.c << 'EOF'
/* Loop Unrolling - 修正版（列優先アクセス） */
void dgemm_unroll(REAL *A, REAL *B, REAL *C, int n)
{
  int i, j, k;
  int unroll_factor = UNROLL;
  
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

/* より効率的なループアンローリング（キャッシュ最適化版） */
void dgemm_unroll_optimized(REAL *A, REAL *B, REAL *C, int n)
{
  int i, j, k;
  int unroll_factor = UNROLL;
  
  // 初期化
  for (i = 0; i < n * n; i++)
    C[i] = 0.0;
  
  // 列優先アクセスパターン（最適化版）
  for (j = 0; j < n; j++)
    for (k = 0; k < n; k++)
    {
      REAL bkj = B[k + j * n];  // B[k][j]を一度読み込み
      // アンロール可能な部分（より大きなアンロール係数）
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
  REAL cij;
  int unroll_factor = UNROLL;

  for (i = si; i < si + BLOCKSIZE; ++i)
  {
    for (j = sj; j < sj + BLOCKSIZE; ++j)
    {
      cij = C[i + j * n];
      for (k = sk; k < sk + BLOCKSIZE - unroll_factor + 1; k += unroll_factor)
      {
        cij += A[i + k * n] * B[k + j * n];
        cij += A[i + (k+1) * n] * B[(k+1) + j * n];
        cij += A[i + (k+2) * n] * B[(k+2) + j * n];
        cij += A[i + (k+3) * n] * B[(k+3) + j * n];
      }
      // 残りの要素を処理
      for (; k < sk + BLOCKSIZE; k++)
        cij += A[i + k * n] * B[k + j * n];
      C[i + j * n] = cij;
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
  REAL cij;
  int unroll_factor = UNROLL;

#pragma omp parallel for private(j,k,cij)
  for (i = 0; i < n; i++)
  {
    for (j = 0; j < n; j++)
    {
      cij = C[i + j * n];
      for (k = 0; k < n - unroll_factor + 1; k += unroll_factor)
      {
        cij += A[i + k * n] * B[k + j * n];
        cij += A[i + (k+1) * n] * B[(k+1) + j * n];
        cij += A[i + (k+2) * n] * B[(k+2) + j * n];
        cij += A[i + (k+3) * n] * B[(k+3) + j * n];
      }
      // 残りの要素を処理
      for (; k < n; k++)
        cij += A[i + k * n] * B[k + j * n];
      C[i + j * n] = cij;
    }
  }
}

EOF

    # ファイルに挿入
    sed -i "${actual_insert_point}a\\" gemm-test.c
    sed -i "${actual_insert_point}r temp_unroll_function.c" gemm-test.c
    
    rm temp_unroll_function.c
    
    print_success "ループアンローリング関数の追加完了"
}

# define文の追加
add_defines() {
    print_info "define文を追加中..."
    
    # 既存のdefine文の後に追加
    sed -i '/\/\/#define AVX_OMP/a\
\/\/#define UNROLL_ONLY                   \/\* Loop Unrolling (Fixed) -> ON *\/\
\/\/#define UNROLL_OPTIMIZED              \/\* Loop Unrolling (Optimized) -> ON *\/\
\/\/#define BLOCKING_UNROLL               \/\* Blocking + Loop Unrolling -> ON *\/\
\/\/#define OMP_UNROLL                    \/\* OpenMP + Loop Unrolling -> ON *\/' gemm-test.c
    
    print_success "define文の追加完了"
}

# main関数にテストコードを追加
add_main_tests() {
    print_info "main関数にテストコードを追加中..."
    
    # MKLテストの後に追加
    local mkl_end=$(grep -n "#endif" gemm-test.c | tail -1 | cut -d: -f1)
    
    cat > temp_main_tests.c << 'EOF'
    /*Loop Unrolling - 修正版 */
#ifdef UNROLL_ONLY
    int_mat(A, B, C, N);
    t = seconds();
    dgemm_unroll(A, B, C, N);
    t = seconds() - t;
    check_mat(C, C_unopt, N);
    printf("%f [s]  GFLOPS %f  |Loop Unrolling (Fixed)|\n", t,
           (float)N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

    /*Loop Unrolling - 最適化版 */
#ifdef UNROLL_OPTIMIZED
    int_mat(A, B, C, N);
    t = seconds();
    dgemm_unroll_optimized(A, B, C, N);
    t = seconds() - t;
    check_mat(C, C_unopt, N);
    printf("%f [s]  GFLOPS %f  |Loop Unrolling (Optimized)|\n", t,
           (float)N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

    /*Blocking + Loop Unrolling */
#ifdef BLOCKING_UNROLL
    int_mat(A, B, C, N);
    t = seconds();
    dgemm_blocking_unroll(A, B, C, N);
    t = seconds() - t;
    check_mat(C, C_unopt, N);
    printf("%f [s]  GFLOPS %f  |Blocking+Unroll|\n", t,
           (float)N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

    /*OpenMP + Loop Unrolling */
#ifdef OMP_UNROLL
    int_mat(A, B, C, N);
    t = seconds();
    dgemm_OMP_unroll(A, B, C, N);
    t = seconds() - t;
    check_mat(C, C_unopt, N);
    printf("%f [s]  GFLOPS %f  |OpenMP+Unroll|\n", t,
           (float)N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

EOF

    # ファイルに挿入
    sed -i "${mkl_end}a\\" gemm-test.c
    sed -i "${mkl_end}r temp_main_tests.c" gemm-test.c
    
    rm temp_main_tests.c
    
    print_success "main関数へのテストコード追加完了"
}

# 実装確認
verify_implementation() {
    print_info "実装の確認中..."
    
    # 必要な関数が存在するかチェック
    local functions=("dgemm_unroll" "dgemm_unroll_optimized" "dgemm_blocking_unroll" "dgemm_OMP_unroll")
    local missing_functions=()
    
    for func in "${functions[@]}"; do
        if ! grep -q "void $func" gemm-test.c; then
            missing_functions+=("$func")
        fi
    done
    
    if [ ${#missing_functions[@]} -eq 0 ]; then
        print_success "全ての関数が正常に追加されました"
    else
        print_warning "以下の関数が見つかりません: ${missing_functions[*]}"
    fi
    
    # define文の確認
    if grep -q "UNROLL_ONLY" gemm-test.c; then
        print_success "define文が正常に追加されました"
    else
        print_warning "define文が見つかりません"
    fi
}

# テスト実行
run_test() {
    print_info "ループアンローリングのテストを実行中..."
    
    # UNROLL_ONLYを有効化
    sed -i 's|//#define UNROLL_ONLY|#define UNROLL_ONLY|' gemm-test.c
    
    make clean
    make
    
    print_info "テスト実行中..."
    make run
    
    # 元に戻す
    sed -i 's|#define UNROLL_ONLY|//#define UNROLL_ONLY|' gemm-test.c
    
    print_success "ループアンローリングテスト完了"
}

# メイン実行
main() {
    print_header "ループアンローリング実装スクリプト"
    
    # バックアップ作成
    create_backup
    
    # 実装
    add_unroll_function
    add_defines
    add_main_tests
    
    # 確認
    verify_implementation
    
    # テスト実行（オプション）
    read -p "ループアンローリングのテストを実行しますか？ (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_test
    fi
    
    print_header "実装完了"
    print_success "ループアンローリングの実装が完了しました"
    print_info "以下のdefine文を使用してテストできます："
    echo "  #define UNROLL_ONLY        // ループアンローリング（修正版）"
    echo "  #define UNROLL_OPTIMIZED   // ループアンローリング（最適化版）"
    echo "  #define BLOCKING_UNROLL    // ブロッキング + ループアンローリング"
    echo "  #define OMP_UNROLL         // OpenMP + ループアンローリング"
}

# ヘルプ表示
show_help() {
    echo "使用方法: $0"
    echo ""
    echo "このスクリプトはgemm-test.cにループアンローリング機能を追加します"
    echo ""
    echo "追加される機能:"
    echo "  - dgemm_unroll: ループアンローリング（修正版）"
    echo "  - dgemm_unroll_optimized: ループアンローリング（最適化版）"
    echo "  - dgemm_blocking_unroll: ブロッキング + ループアンローリング"
    echo "  - dgemm_OMP_unroll: OpenMP + ループアンローリング"
    echo ""
    echo "注意: 実行前にgemm-test.cのバックアップが作成されます"
}

# オプション解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# メイン実行
main "$@"
