# Git Hooks
Git hooks adalah mekanisme dalam Git yang memungkinkan Anda menjalankan skrip otomatis sebagai respons terhadap event tertentu di dalam proses Git, seperti commit, push, atau merge. Dengan Git hooks, Anda dapat memastikan beberapa tindakan tertentu dilakukan sebelum atau setelah event Git, yang bisa membantu dalam menjaga konsistensi dan kualitas kode dalam repository.

Berikut adalah beberapa contoh Git hooks umum:

1. **Pre-commit**: Dijalankan sebelum commit dibuat. Hook ini sering digunakan untuk memeriksa format kode, menghapus data sensitif, atau menjalankan pengujian otomatis. Jika hook ini gagal (misalnya, karena ada tes yang gagal), maka commit tidak akan dilanjutkan.

2. **Pre-push**: Dijalankan sebelum Anda mendorong (push) perubahan ke repository remote. Ini berguna untuk menjalankan pengujian atau pemeriksaan lainnya untuk memastikan bahwa kode yang akan di-push memenuhi standar tertentu.

3. **Post-merge**: Dijalankan setelah proses merge selesai. Biasanya digunakan untuk menjalankan tes atau membersihkan konflik yang terjadi setelah merge.

4. **Pre-rebase**: Dijalankan sebelum rebase dimulai, sering digunakan untuk memastikan kondisi tertentu sebelum memodifikasi riwayat commit.

Git menyimpan skrip hooks ini dalam folder `.git/hooks` di dalam setiap repository. Anda dapat membuat atau memodifikasi file dalam folder ini sesuai kebutuhan dan menggunakan skrip sesuai preferensi (misalnya, Bash, Python, atau bahasa lain).

Git hooks sangat berguna untuk otomatisasi tugas-tugas tertentu dalam workflow Git, terutama dalam menjaga kualitas kode. Git menyediakan dua kategori hooks utama:

1. **Client-side hooks**: Hooks yang berfungsi pada sisi pengguna lokal, misalnya saat melakukan operasi seperti commit, merge, atau push.
2. **Server-side hooks**: Hooks yang berfungsi pada sisi server, yang umumnya digunakan saat melakukan operasi seperti menerima push ke repository remote.

Berikut adalah penjelasan lebih lanjut dari beberapa Git hooks yang umum dan contoh implementasinya:

### 1. `pre-commit` Hook
Hook ini dieksekusi sebelum perubahan direkam dalam commit. Hook ini berguna untuk menjalankan skrip otomatis yang memeriksa atau membersihkan kode sebelum setiap commit. Misalnya, Anda dapat memastikan tidak ada "syntax error" atau format kode yang tidak konsisten.

#### Contoh `pre-commit` Hook
Misalkan, Anda ingin memastikan semua file Python telah melalui linter (misalnya `flake8`) sebelum commit dibuat:

```bash
#!/bin/sh
# pre-commit hook to lint Python files

# Run flake8 on all staged .py files
files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')
if [ "$files" ]; then
    flake8 $files
    if [ $? -ne 0 ]; then
        echo "Linting failed. Please fix the issues and commit again."
        exit 1
    fi
fi
```

Script di atas:
- Mengecek file Python yang di-*staging* menggunakan `git diff --cached`.
- Menjalankan `flake8` untuk memastikan tidak ada kesalahan.
- Jika linter mendeteksi masalah, commit dibatalkan (status `exit 1`).

### 2. `pre-push` Hook
Hook ini dieksekusi sebelum perubahan di-*push* ke repository remote. Umumnya digunakan untuk menjalankan tes atau memastikan bahwa kode memenuhi standar tertentu sebelum dikirimkan ke repository utama.

#### Contoh `pre-push` Hook
Misalnya, Anda ingin menjalankan tes unit sebelum `push`. Jika ada tes yang gagal, `push` akan dibatalkan.

```bash
#!/bin/sh
# Run tests before pushing

echo "Running tests..."
pytest
if [ $? -ne 0 ]; then
    echo "Tests failed. Push aborted."
    exit 1
fi
```

Script di atas:
- Menjalankan `pytest` untuk memverifikasi bahwa semua tes lulus.
- Membatalkan `push` jika ada tes yang gagal.

### 3. `post-merge` Hook
Hook ini dijalankan setelah proses merge selesai. Biasanya digunakan untuk melakukan tugas seperti menginstal dependensi atau membersihkan file sementara.

#### Contoh `post-merge` Hook
Misalkan, Anda ingin memastikan semua dependensi sudah diinstal setiap kali Anda menarik perubahan baru (misalnya, file `requirements.txt` diperbarui):

```bash
#!/bin/sh
# Automatically install dependencies after merge

if [ -f requirements.txt ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi
```

Script di atas:
- Mengecek apakah `requirements.txt` ada.
- Menginstal ulang dependensi yang diperlukan setelah perubahan baru di-*merge* ke branch Anda.

### 4. `prepare-commit-msg` Hook
Hook ini dijalankan setelah commit message diinisialisasi, tetapi sebelum editor terbuka. Berguna untuk menambahkan informasi otomatis ke pesan commit.

#### Contoh `prepare-commit-msg` Hook
Misalnya, Anda ingin secara otomatis menambahkan ID `ticket` JIRA di awal pesan commit untuk setiap perubahan.

```bash
#!/bin/sh
# Auto-prepend JIRA ticket ID to commit messages

ticket_id="[JIRA-123]"
if [ -z "$(grep "$ticket_id" "$1")" ]; then
    echo "$ticket_id $(cat "$1")" > "$1"
fi
```

Script di atas:
- Menambahkan ID `ticket` ke awal pesan commit jika belum ada.
  
### 5. `commit-msg` Hook
Hook ini dieksekusi setelah pesan commit dibuat tetapi sebelum commit disimpan ke Git. Berguna untuk memvalidasi format pesan commit.

#### Contoh `commit-msg` Hook
Misalnya, Anda ingin memastikan semua pesan commit memiliki panjang minimal 10 karakter.

```bash
#!/bin/sh
# Ensure commit messages are at least 10 characters long

commit_msg=$(cat "$1")
msg_len=${#commit_msg}

if [ "$msg_len" -lt 10 ]; then
    echo "Commit message is too short. Please provide a more descriptive message."
    exit 1
fi
```

Script di atas:
- Mengecek panjang pesan commit.
- Membatalkan commit jika pesan kurang dari 10 karakter.

### Cara Mengaktifkan Git Hooks
1. Masuk ke folder `.git/hooks` dalam repository.
2. Tambahkan skrip ke file hook sesuai nama event yang ingin diaktifkan, seperti `pre-commit`, `pre-push`, dll.
3. Pastikan skrip bersifat eksekusi dengan menjalankan `chmod +x <nama-hook>`.

> **Catatan**: Setiap skrip hook harus dimulai dengan shebang (`#!/bin/sh` atau lainnya) dan disimpan tanpa ekstensi file (misalnya `pre-commit`, bukan `pre-commit.sh`). 

Git hooks adalah cara efektif untuk menjaga kualitas kode dan memastikan alur kerja lebih konsisten, terutama dalam tim pengembangan.