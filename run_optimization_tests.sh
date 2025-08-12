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

# 問題の診断
diagnose_issues() {
    print_header "問題の診断"
    
    # 1. defineの状態を確認
    print_info "現在のdefineの状態を確認中..."
    echo "UNROLL_ONLY: $(grep -c '^#define UNROLL_ONLY' gemm-test.c || echo '0')"
    echo "UNROLL_OPTIMIZED: $(grep -c '^#define UNROLL_OPTIMIZED' gemm-test.c || echo '0')"
    echo "BLOCKING_UNROLL: $(grep -c '^#define BLOCKING_UNROLL' gemm-test.c || echo '0')"
    echo "OMP_UNROLL: $(grep -c '^#define OMP_UNROLL' gemm-test.c || echo '0')"
    
    # 2. 関数の存在確認
    print_info "ループアンローリング関数の存在確認中..."
    echo "dgemm_unroll: $(grep -c 'void dgemm_unroll' gemm-test.c || echo '0')"
    echo "dgemm_unroll_optimized: $(grep -c 'void dgemm_unroll_optimized' gemm-test.c || echo '0')"
    
    # 3. main関数のテストコード確認
    print_info "main関数のテストコード確認中..."
    echo "UNROLL_ONLY test block: $(grep -c '#ifdef UNROLL_ONLY' gemm-test.c || echo '0')"
    echo "UNROLL_OPTIMIZED test block: $(grep -c '#ifdef UNROLL_OPTIMIZED' gemm-test.c || echo '0')"
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
    local blocking_count=$(grep -c "void dgemm_blocking_unroll" gemm-test.c || echo "0")
    if [ "$blocking_count" -eq 2 ]; then
        # 2つ目の定義を削除
        sed -i '/^void dgemm_blocking_unroll(REAL \*A, REAL \*B, REAL \*C, int n)$/,/^}$/d' gemm-test.c
        print_success "重複するdgemm_blocking_unroll関数を削除しました"
    fi
    
    # dgemm_OMP_unrollの重複を削除
    local omp_count=$(grep -c "void dgemm_OMP_unroll" gemm-test.c || echo "0")
    if [ "$omp_count" -eq 2 ]; then
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
    sed -i 's|#define UNROLL_OPTIMIZED|//#define UNROLL_OPTIMIZED|' gemm-test.c
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
    
    # 修正版ループアンローリング
    print_info "修正版ループアンローリングテスト"
    sed -i 's|//#define UNROLL_ONLY|#define UNROLL_ONLY|' gemm-test.c
    make clean
    make
    echo "=== 修正版ループアンローリングテスト ==="
    make run
    sed -i 's|#define UNROLL_ONLY|//#define UNROLL_ONLY|' gemm-test.c
    
    # 最適化版ループアンローリング
    print_info "最適化版ループアンローリングテスト"
    sed -i 's|//#define UNROLL_OPTIMIZED|#define UNROLL_OPTIMIZED|' gemm-test.c
    make clean
    make
    echo "=== 最適化版ループアンローリングテスト ==="
    make run
    sed -i 's|#define UNROLL_OPTIMIZED|//#define UNROLL_OPTIMIZED|' gemm-test.c
    
    # ブロッキング + ループアンローリング
    print_info "ブロッキング + ループアンローリングテスト"
    sed -i 's|//#define BLOCKING_UNROLL|#define BLOCKING_UNROLL|' gemm-test.c
    make clean
    make
    echo "=== ブロッキング + ループアンローリングテスト ==="
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
        echo "=== OpenMP + ループアンローリング スレッド数: $threads ==="
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
    local log_file="$1"
    print_header "結果の整理"
    
    print_info "ログファイル: $log_file"
    
    print_info "以下のコマンドで結果を確認できます："
    echo ""
    echo "1. 最高性能の組み合わせを確認:"
    echo "   grep 'GFLOPS' $log_file | sort -k3 -nr | head -10"
    echo ""
    echo "2. 各最適化技法の効果を比較:"
    echo "   grep 'unoptimized\|blocking\|AVX2\|OpenMP\|MKL\|Unroll\|Loop Unrolling' $log_file"
    echo ""
    echo "3. 性能向上率の計算:"
    echo "   基本性能と最高性能を比較して計算してください"
    echo ""
    echo "4. 全結果の表示:"
    echo "   cat $log_file"
    
    # 実際に結果を集計して表示
    print_info "=== 結果集計 ==="
    echo ""
    echo "最高性能トップ10:"
    grep 'GFLOPS' "$log_file" | sort -k3 -nr | head -10
    echo ""
    echo "各最適化技法の平均性能:"
    echo "unoptimized: $(grep 'unoptimized' "$log_file" | awk '{sum+=$3; count++} END {print sum/count " GFLOPS"}')"
    echo "loop exchange: $(grep 'loop exchange' "$log_file" | awk '{sum+=$3; count++} END {print sum/count " GFLOPS"}')"
    echo "blocking: $(grep 'blocking' "$log_file" | awk '{sum+=$3; count++} END {print sum/count " GFLOPS"}')"
    echo "AVX2: $(grep 'AVX2' "$log_file" | awk '{sum+=$3; count++} END {print sum/count " GFLOPS"}')"
    echo "OpenMP: $(grep 'OpenMP' "$log_file" | awk '{sum+=$3; count++} END {print sum/count " GFLOPS"}')"
    echo "MKL: $(grep 'MKL' "$log_file" | awk '{sum+=$3; count++} END {print sum/count " GFLOPS"}')"
    echo "Loop Unrolling: $(grep 'Loop Unrolling' "$log_file" | awk '{sum+=$3; count++} END {print sum/count " GFLOPS"}')"
}

# メイン実行
main() {
    print_header "問題修正とテスト実行"
    
    # ログファイル名を設定
    local log_file="optimization_results_$(date +%Y%m%d_%H%M%S).log"
    
    # 問題の診断
    diagnose_issues
    
    # 問題の修正
    fix_issues
    
    print_info "テスト結果をログファイルに保存します: $log_file"
    
    # 各テストの実行（ログに保存）
    {
        run_basic_test
        run_blocking_tests
        run_avx2_test
        run_openmp_tests
        run_avx2_openmp_tests
        run_unroll_tests
        run_mkl_test
    } 2>&1 | tee "$log_file"
    
    # 結果の整理
    organize_results "$log_file"
    
    print_header "テスト完了"
    print_success "全てのテストが完了しました"
    print_info "結果を確認して最高性能の組み合わせを特定してください"
    print_info "ログファイル: $log_file"
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
