#!/bin/bash

# 行列行列積最適化課題セットアップスクリプト
# 必要なスクリプトに実行権限を付与し、環境をセットアップします

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

# 実行権限の付与
setup_permissions() {
    print_header "実行権限の設定"
    
    local scripts=(
        "run_optimization_tests.sh"
        "implement_unrolling.sh"
        "setup.sh"
    )
    
    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            chmod +x "$script"
            print_success "$script に実行権限を付与しました"
        else
            print_warning "$script が見つかりません"
        fi
    done
}

# ファイル存在確認
check_files() {
    print_header "ファイル存在確認"
    
    local required_files=(
        "gemm-test.c"
        "Makefile"
        "mm.py"
        "README.md"
    )
    
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            print_success "$file が存在します"
        else
            missing_files+=("$file")
            print_error "$file が見つかりません"
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        print_warning "以下のファイルが見つかりません: ${missing_files[*]}"
        return 1
    fi
    
    return 0
}

# 基本コンパイルテスト
test_compilation() {
    print_header "基本コンパイルテスト"
    
    print_info "クリーンビルドを実行中..."
    make clean
    make
    
    if [ -f "gemm-test" ]; then
        print_success "コンパイルが成功しました"
    else
        print_error "コンパイルに失敗しました"
        return 1
    fi
}

# 基本実行テスト
test_execution() {
    print_header "基本実行テスト"
    
    print_info "基本テストを実行中..."
    timeout 30s make run || {
        print_warning "基本テストがタイムアウトしました（正常な場合があります）"
    }
    
    print_success "基本実行テスト完了"
}

# 使用方法の表示
show_usage() {
    print_header "使用方法"
    
    echo "以下のコマンドで課題を実行できます："
    echo ""
    echo "1. ループアンローリングの実装（初回のみ）:"
    echo "   ./implement_unrolling.sh"
    echo ""
    echo "2. 全最適化テストの実行:"
    echo "   ./run_optimization_tests.sh"
    echo ""
    echo "3. 個別テストの実行:"
    echo "   make clean && make && make run"
    echo ""
    echo "4. ヘルプの表示:"
    echo "   ./run_optimization_tests.sh --help"
    echo ""
    echo "5. 手順書の確認:"
    echo "   cat 課題実行手順書.md"
}

# 環境情報の表示
show_environment() {
    print_header "環境情報"
    
    echo "OS: $(uname -a)"
    echo "現在のディレクトリ: $(pwd)"
    echo ""
    
    echo "CPU情報:"
    lscpu | grep "Model name\|CPU(s)\|Thread\|Core" | head -4
    echo ""
    
    echo "コンパイラ情報:"
    if command -v gcc &> /dev/null; then
        echo "GCC: $(gcc --version | head -1)"
    else
        echo "GCC: not found"
    fi
    
    if command -v icc &> /dev/null; then
        echo "ICC: $(icc --version | head -1)"
    else
        echo "ICC: not found"
    fi
    echo ""
    
    echo "OpenMP設定:"
    echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-not set}"
}

# メイン実行
main() {
    print_header "行列行列積最適化課題セットアップ"
    
    # ファイル確認
    if ! check_files; then
        print_error "必要なファイルが見つかりません"
        exit 1
    fi
    
    # 実行権限の設定
    setup_permissions
    
    # 基本コンパイルテスト
    if test_compilation; then
        print_success "セットアップが完了しました"
    else
        print_error "コンパイルテストに失敗しました"
        exit 1
    fi
    
    # 環境情報の表示
    show_environment
    
    # 使用方法の表示
    show_usage
    
    print_header "セットアップ完了"
    print_success "課題の実行準備が完了しました"
    print_info "まず ./implement_unrolling.sh を実行してループアンローリングを実装してください"
}

# ヘルプ表示
show_help() {
    echo "使用方法: $0"
    echo ""
    echo "このスクリプトは行列行列積最適化課題の実行環境をセットアップします"
    echo ""
    echo "実行内容:"
    echo "  - 必要なファイルの存在確認"
    echo "  - スクリプトへの実行権限付与"
    echo "  - 基本コンパイルテスト"
    echo "  - 環境情報の表示"
    echo "  - 使用方法の表示"
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
