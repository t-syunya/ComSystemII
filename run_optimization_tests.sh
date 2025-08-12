#!/bin/bash

# 問題修正とテスト実行スクリプト
# 発生した問題を修正して正しくテストを実行します

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

# 問題の修正
fix_issues() {
    print_header "問題の修正"
    
    # 1. BLOCKSIZEを32に戻す
    print_info "BLOCKSIZEを32に戻しています..."
    sed -i 's/#define BLOCKSIZE 128/#define BLOCKSIZE 32/' gemm-test.c
    sed -i 's/#define BLOCKSIZE 64/#define BLOCKSIZE 32/' gemm-test.c
    sed -i 's/#define BLOCKSIZE 16/#define BLOCKSIZE 32/' gemm-test.c
    print_success "BLOCKSIZEを32に戻しました"
    
    # 2. 重複関数の削除
    print_info "重複関数を削除しています..."
    
    # dgemm_blocking_unrollの重複を削除
    if grep -n "void dgemm_blocking_unroll" gemm-test.c | wc -l | grep -q "2"; then
        # 2つ目の定義を削除
        sed -i '/^void dgemm_blocking_unroll(REAL \*A, REAL \*B, REAL \*C, int n)$/,/^}$/d' gemm-test.c
        print_success "重複するdgemm_blocking_unroll関数を削除しました"
    fi
    
    # dgemm_OMP_unrollの重複を削除
    if grep -n "void dgemm_OMP_unroll" gemm-test.c | wc -l | grep -q "2"; then
        # 2つ目の定義を削除
        sed -i '/^void dgemm_OMP_unroll(REAL \*A, REAL \*B, REAL \*C, int n)$/,/^}$/d' gemm-test.c
        print_success "重複するdgemm_OMP_unroll関数を削除しました"
    fi
    
    # 3. 全てのdefineを無効化
    print_info "全てのdefineを無効化しています..."
    sed -i 's|#define BLOCKING|//#define BLOCKING|' gemm-test.c
    sed -i 's|#define AVX2|//#define AVX2|' gemm-test.c
    sed -i 's|#define OMP|//#define OMP|' gemm-test.c
    sed -i 's|#define AVX_OMP|//#define AVX_OMP|' gemm-test.c
    sed -i 's|#define MKL|//#define MKL|' gemm-test.c
    sed -i 's|#define UNROLL_ONLY|//#define UNROLL_ONLY|' gemm-test.c
    sed -i 's|#define BLOCKING_UNROLL|//#define BLOCKING_UNROLL|' gemm-test.c
    sed -i 's|#define OMP_UNROLL|//#define OMP_UNROLL|' gemm-test.c
    print_success "全てのdefineを無効化しました"
}

# 基本テスト
run_basic_test() {
    print_header "基本テスト（最適化なし）"
    
    make clean
    make
    
    print_info "基本テスト実行中..."
    make run
    
    print_success "基本テスト完了"
}

# ブロッキングテスト（修正版）
run_blocking_tests() {
    print_header "ブロッキングテスト"
    
    local block_sizes=(16 32 64)
    
    for block_size in "${block_sizes[@]}"; do
        print_info "ブロックサイズ: $block_size"
        
        # BLOCKSIZEを変更
        sed -i "s/#define BLOCKSIZE [0-9]*/#define BLOCKSIZE $block_size/" gemm-test.c
        sed -i 's|//#define BLOCKING|#define BLOCKING|' gemm-test.c
        
        make clean
        make
        
        print_info "ブロッキングテスト実行中（ブロックサイズ: $block_size）..."
        make run
        
        # 元に戻す
        sed -i 's|#define BLOCKING|//#define BLOCKING|' gemm-test.c
        
        print_success "ブロックサイズ $block_size のテスト完了"
    done
    
    # BLOCKSIZEを32に戻す
    sed -i "s/#define BLOCKSIZE [0-9]*/#define BLOCKSIZE 32/" gemm-test.c
}

# AVX2テスト
run_avx2_test() {
    print_header "AVX2ベクトル化テスト"
    
    sed -i 's|//#define AVX2|#define AVX2|' gemm-test.c
    make clean
    make
    
    print_info "AVX2テスト実行中..."
    make run
    
    # 元に戻す
    sed -i 's|#define AVX2|//#define AVX2|' gemm-test.c
    
    print_success "AVX2テスト完了"
}

# OpenMPテスト
run_openmp_tests() {
    print_header "OpenMP並列化テスト"
    
    local thread_counts=(1 2 4 8)
    
    sed -i 's|//#define OMP|#define OMP|' gemm-test.c
    make clean
    make
    
    for threads in "${thread_counts[@]}"; do
        print_info "スレッド数: $threads"
        
        export OMP_NUM_THREADS=$threads
        
        print_info "OpenMPテスト実行中（スレッド数: $threads）..."
        make run
        
        print_success "スレッド数 $threads のテスト完了"
    done
    
    # 元に戻す
    sed -i 's|#define OMP|//#define OMP|' gemm-test.c
}

# AVX2 + OpenMPテスト
run_avx2_openmp_tests() {
    print_header "AVX2 + OpenMP組み合わせテスト"
    
    local thread_counts=(1 2 4 8)
    
    sed -i 's|//#define AVX_OMP|#define AVX_OMP|' gemm-test.c
    make clean
    make
    
    for threads in "${thread_counts[@]}"; do
        print_info "AVX2 + OpenMP スレッド数: $threads"
        
        export OMP_NUM_THREADS=$threads
        
        print_info "AVX2 + OpenMPテスト実行中（スレッド数: $threads）..."
        make run
        
        print_success "AVX2 + OpenMP スレッド数 $threads のテスト完了"
    done
    
    # 元に戻す
    sed -i 's|#define AVX_OMP|//#define AVX_OMP|' gemm-test.c
}

# ループアンローリングテスト
run_unroll_tests() {
    print_header "ループアンローリングテスト"
    
    # 基本的なループアンローリング
    print_info "基本的なループアンローリングテスト"
    sed -i 's|//#define UNROLL_ONLY|#define UNROLL_ONLY|' gemm-test.c
    make clean
    make
    make run
    sed -i 's|#define UNROLL_ONLY|//#define UNROLL_ONLY|' gemm-test.c
    
    # ブロッキング + ループアンローリング
    print_info "ブロッキング + ループアンローリングテスト"
    sed -i 's|//#define BLOCKING_UNROLL|#define BLOCKING_UNROLL|' gemm-test.c
    make clean
    make
    make run
    sed -i 's|#define BLOCKING_UNROLL|//#define BLOCKING_UNROLL|' gemm-test.c
    
    # OpenMP + ループアンローリング
    print_info "OpenMP + ループアンローリングテスト"
    sed -i 's|//#define OMP_UNROLL|#define OMP_UNROLL|' gemm-test.c
    make clean
    make
    
    # 様々なスレッド数でテスト
    for threads in 1 2 4 8; do
        print_info "OpenMP + ループアンローリング スレッド数: $threads"
        export OMP_NUM_THREADS=$threads
        make run
    done
    
    sed -i 's|#define OMP_UNROLL|//#define OMP_UNROLL|' gemm-test.c
    
    print_success "ループアンローリングテスト完了"
}

# MKLテスト
run_mkl_test() {
    print_header "MKLライブラリテスト"
    
    sed -i 's|//#define MKL|#define MKL|' gemm-test.c
    make clean
    make
    
    print_info "MKLテスト実行中..."
    make run
    
    # 元に戻す
    sed -i 's|#define MKL|//#define MKL|' gemm-test.c
    
    print_success "MKLテスト完了"
}

# 結果の整理
organize_results() {
    print_header "結果の整理"
    
    print_info "以下のコマンドで結果を確認できます："
    echo ""
    echo "1. 最高性能の組み合わせを確認:"
    echo "   grep 'GFLOPS' /dev/stdout | sort -k3 -nr | head -10"
    echo ""
    echo "2. 各最適化技法の効果を比較:"
    echo "   grep 'unoptimized\|blocking\|AVX2\|OpenMP\|MKL\|Unroll' /dev/stdout"
    echo ""
    echo "3. 性能向上率の計算:"
    echo "   基本性能と最高性能を比較して計算してください"
}

# メイン実行
main() {
    print_header "問題修正とテスト実行"
    
    # 問題の修正
    fix_issues
    
    # 各テストの実行
    run_basic_test
    run_blocking_tests
    run_avx2_test
    run_openmp_tests
    run_avx2_openmp_tests
    run_unroll_tests
    run_mkl_test
    
    # 結果の整理
    organize_results
    
    print_header "テスト完了"
    print_success "全てのテストが完了しました"
    print_info "結果を確認して最高性能の組み合わせを特定してください"
}

# ヘルプ表示
show_help() {
    echo "使用方法: $0"
    echo ""
    echo "このスクリプトは発生した問題を修正して正しくテストを実行します"
    echo ""
    echo "修正内容:"
    echo "  - BLOCKSIZEの問題を修正"
    echo "  - 重複関数の削除"
    echo "  - ループアンローリングテストの実行"
    echo "  - 各最適化技法の正しいテスト"
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
