#!/bin/bash

# ==========================================
# CẤU HÌNH THÔNG TIN
# ==========================================
TARGET="/home/dell/besttrack/shp/"
COMMIT_MSG="Cập nhật dữ liệu không gian: $TARGET"
BRANCH="main"
PRIVATE_KEY_PATH="$HOME/.ssh/id_ed25519"
SIZE_LIMIT=$((100 * 1024 * 1024))

# ==========================================
# PHẦN THỰC THI
# ==========================================
echo "Bat dau qua trinh day du lieu len GitHub..."

# 1. SSH key
export GIT_SSH_COMMAND="ssh -i $PRIVATE_KEY_PATH -o IdentitiesOnly=yes"

# 2. Kiem tra git-lfs
if ! command -v git-lfs &> /dev/null; then
    echo "THAT BAI: git-lfs chua duoc cai dat! Chay: sudo apt install git-lfs"
    exit 1
fi

# 3. Thiet lap Git LFS
echo "[1/7] Thiet lap Git LFS..."
git lfs install > /dev/null 2>&1
git lfs track "*.shp" > /dev/null 2>&1
git lfs track "*.dbf" > /dev/null 2>&1
git lfs track "*.shx" > /dev/null 2>&1
git lfs track "*.nc"  > /dev/null 2>&1
git lfs track "*.tif" > /dev/null 2>&1
git lfs track "*.geojson" > /dev/null 2>&1
git add .gitattributes
git diff --cached --quiet .gitattributes || git commit -m "chore: cap nhat Git LFS tracking rules" > /dev/null 2>&1

# 4. Migrate file lon trong lich su sang LFS
echo "[2/7] Migrate file lon trong lich su sang LFS..."
git lfs migrate import \
    --include="*.shp,*.dbf,*.shx,*.nc,*.tif,*.geojson" \
    --include-ref="refs/heads/$BRANCH" \
    --yes > /dev/null 2>&1 \
    && echo "      OK: Migrate hoan tat." \
    || echo "      OK: Khong co file nao can migrate."

# 5. Go loi rebase dang ket
if [ -d ".git/rebase-merge" ] || [ -d ".git/rebase-apply" ]; then
    echo "      Phat hien Git dang ket rebase, tu dong don dep..."
    git rebase --abort > /dev/null 2>&1
fi

# 6. Dong bo tu GitHub — giu ban local neu conflict
echo "[3/7] Dong bo du lieu moi nhat tu GitHub..."
git fetch origin "$BRANCH" > /dev/null 2>&1

# Merge, neu conflict thi tu dong giu ban LOCAL (ours)
if ! git merge origin/"$BRANCH" --no-edit > /dev/null 2>&1; then
    echo "      Phat hien conflict, tu dong giu ban local..."
    git checkout --ours . > /dev/null 2>&1
    git add . > /dev/null 2>&1
    git commit -m "fix: giai quyet conflict, giu ban local" > /dev/null 2>&1
    echo "      OK: Da giai quyet conflict."
fi

# 7. Kiem tra file >100MB chua qua LFS
echo "[4/7] Kiem tra kich thuoc file..."
LARGE_FILES_FOUND=0
while IFS= read -r -d '' file; do
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
    if [ "$size" -gt "$SIZE_LIMIT" ]; then
        if ! (head -c 200 "$file" | grep -q "version https://git-lfs"); then
            echo "      CANH BAO: File lon chua qua LFS: $file ($(( size / 1024 / 1024 )) MB)"
            LARGE_FILES_FOUND=1
        fi
    fi
done < <(find "$TARGET" -type f -print0 2>/dev/null)

if [ "$LARGE_FILES_FOUND" -eq 1 ]; then
    echo "THAT BAI: Co file >100MB chua duoc LFS quan ly."
    echo "   Chay thu cong: git lfs migrate import --include=\"*.shp\" --include-ref=refs/heads/$BRANCH"
    exit 1
fi

# 8. Add file
echo "[5/7] Dang add file/thu muc..."
git add "$TARGET"

# 9. Commit
echo "[6/7] Dang dong goi (commit)..."
git commit -m "$COMMIT_MSG" || echo "      Khong co thay doi nao moi de dong goi."

# 10. Push
echo "[7/7] Dang day len GitHub (nhanh $BRANCH)..."
echo "      (Co the mat vai phut tuy dung luong file LFS)"

git push origin "$BRANCH" --force-with-lease

if [ $? -eq 0 ]; then
    echo ""
    echo "THANH CONG! Da day du lieu len GitHub an toan."
else
    echo ""
    echo "THAT BAI! Goi y kiem tra:"
    echo "   - Ket noi mang / SSH key"
    echo "   - GitHub LFS quota: https://github.com/settings/billing"
    echo "   - Chay thu cong: git lfs status"
fi
