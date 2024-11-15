from spleeter.separator import Separator

# Khởi tạo bộ tách nguồn âm thanh
separator = Separator('spleeter:2stems')

# Tách âm thanh vào các thành phần
sources = separator.separate('ltkt.mp3')

# Lưu các thành phần tách biệt
separator.save_stems('ex', sources)