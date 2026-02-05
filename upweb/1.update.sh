#!/bin/bash

PROJECT_DIR="/home/lephong/besttrack"
# Đảm bảo bạn thay đúng username GitHub của mình vào đây nếu muốn tự động hóa hoàn toàn
GITHUB_USER="phong-levan" 
REPO_NAME="besttrack"

cd "$PROJECT_DIR" || exit

# 1. Tự động tạo file .gitignore để tránh đẩy file rác (venv, lib, bin...)
cat <<EOF > .gitignore
venv/
besttrack_env/
bin/
lib/
lib64/
share/
pyvenv.cfg
__pycache__/
*.pyc
EOF

# 2. Khởi tạo Git nếu chưa có
if [ ! -d ".git" ]; then
    git init
    git branch -M main
    echo "--- Đã khởi tạo Git Local ---"
fi

# 3. Kiểm tra và thiết lập địa chỉ GitHub (Remote)
REMOTE_URL=$(git remote get-url origin 2>/dev/null)

if [ -z "$REMOTE_URL" ]; then
    echo "--- Chưa có địa chỉ GitHub. Vui lòng dán link SSH vào đây ---"
    echo "(Ví dụ: git@github.com:$GITHUB_USER/$REPO_NAME.git)"
    read -p "Link SSH: " NEW_URL
    git remote add origin "$NEW_URL"
else
    echo "--- Địa chỉ hiện tại: $REMOTE_URL ---"
    read -p "Bạn có muốn cập nhật lại địa chỉ mới không? (y/n): " choice
    if [ "$choice" == "y" ]; then
        read -p "Nhập link SSH mới: " NEW_URL
        git remote set-url origin "$NEW_URL"
    fi
fi

# 4. Gom dữ liệu và Commit
git add ve.py besttrack.xlsx icon/ .gitignore
current_time=$(date "+%Y-%m-%d %H:%M:%S")
git commit -m "Cập nhật dữ liệu bão lúc $current_time"

# 5. Đẩy dữ liệu lên GitHub
echo "--- Đang đẩy dữ liệu lên GitHub ---"
if git push -u origin main; then
    echo "--- THÀNH CÔNG! Kiểm tra tại: https://github.com/$GITHUB_USER/$REPO_NAME ---"
else
    echo "--- LỖI: Không thể đẩy dữ liệu. Kiểm tra lại tên Repository trên Web GitHub ---"
fi