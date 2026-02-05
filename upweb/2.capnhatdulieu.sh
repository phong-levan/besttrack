cd /home/lephong/besttrack

# 1. Xóa địa chỉ cũ cho chắc chắn
git remote remove origin

# 2. Thêm lại địa chỉ chuẩn (Thay 'phong-levan' nếu username thực tế của bạn khác)
git remote add origin git@github.com:phong-levan/besttrack.git

# 3. Đẩy dữ liệu lên
git push -u origin main
