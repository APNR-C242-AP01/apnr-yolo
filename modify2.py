import pandas as pd
import re
from difflib import get_close_matches
from collections import Counter

# Fungsi untuk membersihkan karakter yang tidak diinginkan dari plat nomor
def clean_license_number(plate):
    if isinstance(plate, str):  # Pastikan input adalah string
        # Hapus spasi dan karakter yang tidak perlu
        return re.sub(r'[^A-Z0-9]', '', plate.upper())  # Menjaga hanya huruf dan angka
    return plate

# Fungsi untuk memeriksa apakah plat nomor sesuai format Indonesia
def is_valid_plate(plate):
    pattern = r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$'
    return bool(re.match(pattern, plate))

# Fungsi untuk menghitung jumlah karakter yang cocok antara dua string tanpa memperhatikan posisi
def count_matching_chars(str1, str2):
    # Hapus spasi dari kedua string sebelum perbandingan
    str1 = str1.replace(" ", "")
    str2 = str2.replace(" ", "")
    
    # Menggunakan Counter untuk menghitung kemunculan karakter
    counter1 = Counter(str1)
    counter2 = Counter(str2)
    
    # Hitung jumlah karakter yang sama
    common_chars = sum((counter1 & counter2).values())
    
    return common_chars

# Fungsi untuk menemukan plat nomor valid yang memiliki kesamaan karakter minimal 3 karakter
def find_closest_valid_plate(invalid_plate, valid_plates):
    if not isinstance(invalid_plate, str):
        return None  # Abaikan jika bukan string
    
    # Bersihkan plat nomor yang invalid
    invalid_plate = clean_license_number(invalid_plate)
    
    # Cari plat nomor valid dengan angka yang mirip
    candidates = [plate for plate in valid_plates if re.search(r'\d+', plate)]
    
    # Periksa setiap kandidat plat nomor
    closest_plate = None
    max_matching_chars = 0
    for candidate in candidates:
        matching_chars = count_matching_chars(invalid_plate, candidate)
        # Ganti plat nomor jika ada minimal 3 karakter yang cocok
        if matching_chars >= 3 and matching_chars > max_matching_chars:
            max_matching_chars = matching_chars
            closest_plate = candidate
    
    # Kembalikan plat nomor yang paling cocok jika ada, jika tidak ada, kembalikan plat asli
    return closest_plate if closest_plate else invalid_plate

# Fungsi untuk memuat kode region dari file
def load_region(file_path):
    region_codes = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  
                code, region = line.strip().split(': ')
                region_codes[code] = region
    return region_codes

# Fungsi untuk mendapatkan region dari kode plat nomor
def get_region_from_plate(plate, region_codes):
    if not isinstance(plate, str):
        return None
    # Bersihkan dan ganti angka dengan huruf jika perlu
    plate = clean_license_number(plate)
    # Ambil kode huruf awal (1 atau 2 huruf)
    match = re.match(r'^([A-Z]{1,2})', plate)
    if match:
        code = match.group(1)
        # Cari kode region terdekat
        region_code = region_codes.get(code)
        return region_code if region_code else 'Unknown'
    return 'Unknown'

# Proses DataFrame untuk memperbaiki plat nomor dan menambahkan region
def process_license_numbers(df, region_codes):
    # Bersihkan data di kolom license_number
    df['license_number'] = df['license_number'].apply(clean_license_number)

    # Kumpulkan semua plat nomor valid
    valid_plates = df['license_number'].dropna().unique()
    valid_plates = [plate for plate in valid_plates if is_valid_plate(str(plate))]

    # Perbaiki plat nomor tidak valid
    for idx, row in df.iterrows():
        plate = str(row['license_number'])
        if not is_valid_plate(plate):
            closest_plate = find_closest_valid_plate(plate, valid_plates)
            if closest_plate:
                df.at[idx, 'license_number'] = closest_plate

    # Setelah plat nomor diperbaiki, tentukan region
    df['region'] = df['license_number'].apply(lambda x: get_region_from_plate(x, region_codes))

    return df

# Fungsi untuk menambahkan spasi sesuai format plat nomor Indonesia
def format_with_space(plate):
    if isinstance(plate, str):
        # Pisahkan dengan regex sesuai format plat Indonesia
        match = re.match(r'^([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})$', plate)
        if match:
            return f"{match.group(1)} {match.group(2)} {match.group(3)}"
    return plate

if __name__ == "__main__":
    # Membaca file CSV dan kode region
    input_file = 'hasil_plat_indo.csv'
    output_file = 'modify.csv'
    region_file = 'region.txt'

    # Load kode region
    REGION_CODES = load_region(region_file)

    df = pd.read_csv(input_file, on_bad_lines='skip')

    # Proses plat nomor
    df_fixed = process_license_numbers(df, REGION_CODES)

    # Tambahkan spasi pada kolom 'license_number' setelah semua proses selesai
    df_fixed['license_number'] = df_fixed['license_number'].apply(format_with_space)

    # Simpan hasil ke file CSV baru
    df_fixed.to_csv(output_file, index=False)

    # Tampilkan hasil
    print(df_fixed.head())

