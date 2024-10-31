# Registers

Dalam **system programming**, terutama ketika berinteraksi dengan perangkat keras atau menulis program yang sangat efisien seperti dalam **bahasa assembly** atau **bahasa C**, istilah **register** mengacu pada lokasi penyimpanan kecil dan sangat cepat yang terdapat di dalam **CPU**. Register digunakan untuk menyimpan data sementara saat CPU melakukan operasi seperti perhitungan, pengambilan instruksi, atau manipulasi data.

**General-Purpose Registers (GPR)**

| **eax** | Accumulator                                                              |
| ------- | ------------------------------------------------------------------------ |
| **ebx** | Base register                                                            |
| **ecx** | Counter register                                                         |
| **edx** | Data register - can be used for I/O port access and arithmetic functions |
| **esi** | Source index register                                                    |
| **edi** | Destination index register                                               |
| **ebp** | Base pointer register                                                    |
| **esp** | Stack pointer                                                            |

- **EAX**: Akumulator untuk operasi aritmatika dan logika. Sering digunakan untuk menyimpan hasil operasi.
- **EBX**: Biasanya digunakan sebagai base pointer untuk pengalamatan memori.
- **ECX**: Register penghitung, sering digunakan untuk loop dan operasi pengulangan.
- **EDX**: Data register, digunakan untuk menyimpan nilai yang lebih besar dalam operasi pembagian dan perkalian.
- **ESI/EDI**: Source Index dan Destination Index, digunakan untuk operasi terkait string.
- **ESP**: Stack Pointer, menunjuk ke posisi teratas di stack.
- **EBP**: Base Pointer, digunakan untuk menyimpan basis dari stack frame selama pemanggilan fungsi.

| 64-bit register | Lower 32 bits | Lower 16 bits | Lower 8 bits |
| --------------- | ------------- | ------------- | ------------ |
| rax             | eax           | ax            | al           |
| rbx             | ebx           | bx            | bl           |
| rcx             | ecx           | cx            | cl           |
| rdx             | edx           | dx            | dl           |
| rsi             | esi           | si            | sil          |
| rdi             | edi           | di            | dil          |
| rbp             | ebp           | bp            | bpl          |
| rsp             | esp           | sp            | spl          |
| r8              | r8d           | r8w           | r8b          |
| r9              | r9d           | r9w           | r9b          |
| r10             | r10d          | r10w          | r10b         |
| r11             | r11d          | r11w          | r11b         |
| r12             | r12d          | r12w          | r12b         |
| r13             | r13d          | r13w          | r13b         |
| r14             | r14d          | r14w          | r14b         |
| r15             | r15d          | r15w          | r15b         |

### Cara Kerja Register
1. **Kecepatan Tinggi**: Register memiliki kecepatan yang jauh lebih tinggi dibandingkan dengan memori utama (RAM) karena berada langsung di dalam CPU. Oleh karena itu, data yang sering digunakan atau operasi yang memerlukan kecepatan tinggi biasanya disimpan di register.
   
2. **Akses Langsung oleh CPU**: Register diakses langsung oleh unit pemrosesan CPU selama eksekusi instruksi. Misalnya, ketika instruksi `add` dieksekusi, CPU akan mengambil nilai dari register tertentu, menambahkannya, dan menyimpan hasilnya di register.

3. **Ukuran Terbatas**: Jumlah dan ukuran register sangat terbatas dibandingkan dengan memori utama. Setiap CPU memiliki arsitektur yang berbeda dengan jumlah register yang berbeda. Misalnya, prosesor **x86** memiliki register umum seperti `EAX`, `EBX`, `ECX`, `EDX`, dan lain-lain.

4. **Tipe Register**:
   - **General-purpose registers**: Digunakan untuk menyimpan data sementara selama perhitungan.
   - **Special-purpose registers**: Digunakan untuk menyimpan informasi status CPU, seperti **Instruction Pointer (IP)** yang melacak alamat instruksi berikutnya yang akan dijalankan, atau **Stack Pointer (SP)** yang melacak posisi teratas dari stack.
   - **Floating-point registers**: Untuk perhitungan bilangan desimal atau floating-point.
   
5. **Arsitektur CPU**: Register memiliki ukuran yang bervariasi tergantung pada arsitektur CPU. Sebagai contoh:
   - Pada arsitektur **32-bit**, ukuran register umum adalah **32-bit**.
   - Pada arsitektur **64-bit**, ukuran register adalah **64-bit**.

### Contoh dalam System Programming

Dalam pemrograman bahasa tingkat rendah seperti **C** atau **Assembly**, kita sering kali menggunakan register secara langsung atau tidak langsung. Berikut adalah contoh penggunaan register dalam **Assembly**:

```assembly
mov eax, 5       ; Memuat nilai 5 ke dalam register eax
add eax, 3       ; Menambahkan 3 ke nilai yang ada di register eax
mov ebx, eax     ; Memindahkan nilai dari eax ke register ebx
```

Dalam kode di atas:
- `eax` dan `ebx` adalah register, dan instruksi-instruksi ini mengoperasikan data yang disimpan di dalamnya.

### Mengapa Register Penting?
1. **Efisiensi**: Karena register sangat cepat diakses, penggunaan register secara efektif dapat sangat meningkatkan performa program, terutama dalam operasi yang intensif secara komputasional.
2. **Pengoptimalan Kompiler**: Saat menulis kode tingkat tinggi, kompiler akan mencoba mengalokasikan variabel-variabel yang sering digunakan ke dalam register jika memungkinkan, untuk meningkatkan efisiensi.
3. **Kendali Rendah**: Dalam **system programming**, menggunakan register secara langsung memberi kontrol lebih baik atas cara kerja CPU, yang penting dalam pengembangan kernel, driver, atau aplikasi performa tinggi lainnya.

# Opcode

**Opcode** (singkatan dari **operation code**) adalah bagian dari instruksi mesin yang menentukan operasi spesifik yang harus dilakukan oleh **CPU**. Setiap instruksi dalam bahasa mesin terdiri dari dua bagian utama:
1. **Opcode**: Menyatakan operasi apa yang akan dilakukan (misalnya, penjumlahan, pengurangan, penyimpanan data, dll.).
2. **Operand**: Menyatakan data atau alamat yang menjadi target operasi tersebut (bisa berupa register, alamat memori, atau nilai langsung).

### Cara Kerja Opcode
Ketika CPU membaca instruksi dari memori, ia memecah instruksi tersebut menjadi **opcode** dan **operand**. **Opcode** memberi tahu CPU tindakan apa yang harus dilakukan, sedangkan **operand** memberi tahu CPU di mana mendapatkan data yang diperlukan atau ke mana harus menempatkan hasilnya.

Sebagai contoh, dalam instruksi assembly berikut:

```assembly
mov eax, 5
```

- **Opcode**: `mov`, yang menyatakan bahwa CPU harus memindahkan data dari satu lokasi ke lokasi lain.
- **Operand**: `eax` dan `5`, di mana `eax` adalah register tujuan dan `5` adalah nilai yang akan dipindahkan.

Opcode adalah bagian yang diterjemahkan oleh CPU menjadi serangkaian operasi tingkat rendah yang dijalankan secara langsung oleh perangkat keras.

### Opcode dalam Instruksi Mesin
Opcode direpresentasikan sebagai **kode biner** dalam instruksi mesin yang dipahami oleh CPU. Misalnya, pada prosesor x86, instruksi `mov eax, 5` akan diterjemahkan ke dalam kode biner yang berisi **opcode** dan **operand** yang relevan. Contoh sederhana:
- Dalam instruksi mesin, **opcode** bisa saja berupa **nilai biner** seperti `10110000`, yang bisa mewakili instruksi "mov" untuk memindahkan data ke register.

### Contoh Penggunaan Opcode
Misalkan dalam arsitektur **x86**:
- Opcode `ADD` mungkin diterjemahkan sebagai `00000001`, yang menyuruh CPU untuk menambahkan nilai dari dua register.
- Opcode `SUB` mungkin diterjemahkan sebagai `00101000`, yang menyuruh CPU untuk mengurangi satu nilai dari nilai lain.

Dalam bahasa assembly, instruksi ini tampak sederhana seperti:
```assembly
add eax, ebx    ; Menambahkan isi register ebx ke eax
sub eax, 1      ; Mengurangi 1 dari isi register eax
```

Tetapi setelah dikompilasi, instruksi ini diterjemahkan menjadi kode biner, dengan opcode dan operand masing-masing ditentukan secara spesifik untuk CPU.

### Hubungan antara Opcode dan Instruction Set Architecture (ISA)
Opcode bergantung pada **Instruction Set Architecture (ISA)** dari CPU yang digunakan. ISA adalah kumpulan instruksi yang didukung oleh sebuah arsitektur prosesor. Misalnya:
- Pada arsitektur **x86**, ada serangkaian opcode yang spesifik untuk instruksi-instruksi seperti `MOV`, `ADD`, `SUB`, `MUL`, dan lain-lain.
- Pada arsitektur **ARM**, opcode dan format instruksinya akan berbeda meskipun konsep dasarnya sama.

### Pentingnya Opcode dalam System Programming
1. **Kendali Rendah atas CPU**: Menggunakan opcode dalam pemrograman tingkat rendah (misalnya, dalam assembly) memungkinkan pemrogram untuk mengendalikan operasi CPU dengan sangat detail.
2. **Efisiensi Eksekusi**: Karena opcode diterjemahkan langsung menjadi sinyal listrik yang mengendalikan komponen internal CPU, ini adalah cara paling efisien untuk mengeksekusi instruksi.
3. **Pembuatan Kompiler**: Saat menulis kode tingkat tinggi (misalnya, dalam C atau C++), kompiler menerjemahkan instruksi tingkat tinggi menjadi kode mesin yang mengandung opcode CPU, sehingga kode tersebut dapat dijalankan secara langsung oleh perangkat keras.

# Sytem Call (syscall)
**System call (syscall)** adalah mekanisme yang memungkinkan program aplikasi meminta layanan atau sumber daya dari **sistem operasi** (OS). System call memungkinkan program berinteraksi langsung dengan kernel sistem operasi untuk melakukan operasi penting seperti mengakses file, mengelola memori, memulai proses baru, atau berkomunikasi dengan perangkat keras. Dengan menggunakan system call, aplikasi dapat meminta OS untuk menjalankan tugas-tugas yang tidak bisa langsung diakses oleh program pengguna karena keterbatasan keamanan atau kontrol perangkat keras.

### Mengapa System Call Diperlukan?
Sistem operasi mengatur semua sumber daya perangkat keras (CPU, memori, perangkat I/O) dan memastikan bahwa tidak ada program pengguna yang dapat merusak stabilitas sistem atau mengakses sumber daya secara tidak sah. Untuk alasan keamanan, akses langsung ke perangkat keras atau operasi tingkat rendah tidak diizinkan oleh program aplikasi biasa. System call menyediakan "jembatan" yang aman antara program aplikasi dan kernel.

### Cara Kerja System Call
1. **Program Aplikasi Membuat Permintaan**: Ketika sebuah program membutuhkan layanan dari sistem operasi (misalnya, untuk membaca file), ia membuat **system call**.
   
2. **Transisi dari Mode User ke Mode Kernel**: System call menyebabkan transisi dari **user mode** (di mana aplikasi berjalan) ke **kernel mode** (di mana sistem operasi menjalankan operasi yang lebih aman dan kritis).
   
3. **Kernel Menangani Permintaan**: Kernel menangkap system call dan memprosesnya. Kernel memiliki izin untuk berinteraksi dengan perangkat keras dan sistem operasi pada level rendah.

4. **Hasil Dikirim Kembali ke Aplikasi**: Setelah kernel menyelesaikan permintaan, hasilnya dikembalikan ke program aplikasi, dan CPU kembali ke user mode.

### Contoh System Call
Beberapa system call umum di Linux/Unix yang sering digunakan oleh program aplikasi:

1. **File Management**:
   - `open()`: Membuka file.
   - `read()`: Membaca data dari file.
   - `write()`: Menulis data ke file.
   - `close()`: Menutup file setelah selesai digunakan.

2. **Proses Management**:
   - `fork()`: Membuat proses baru dengan menduplikasi proses saat ini.
   - `exec()`: Menjalankan program baru dalam konteks proses yang ada.
   - `wait()`: Menunggu proses anak selesai.

3. **Memory Management**:
   - `mmap()`: Memetakan file atau perangkat ke dalam memori.
   - `brk()`: Menambah atau mengurangi ukuran area data dari proses.

4. **Device Management**:
   - `ioctl()`: Mengirimkan perintah ke perangkat I/O.

5. **Networking**:
   - `socket()`: Membuat soket jaringan.
   - `bind()`: Mengaitkan alamat ke soket.
   - `listen()`: Menunggu koneksi di soket.
   - `accept()`: Menerima koneksi masuk di soket.

### Contoh Penggunaan System Call di C
Berikut adalah contoh penggunaan system call dalam bahasa C untuk **membaca file** menggunakan `read()` dan `write()`:

```c
#include <unistd.h>
#include <fcntl.h>

int main() {
    int file_desc;
    char buffer[100];
    
    // Membuka file "example.txt" dalam mode read-only
    file_desc = open("example.txt", O_RDONLY);
    if (file_desc < 0) {
        // Jika file tidak bisa dibuka
        return 1;
    }
    
    // Membaca isi file ke dalam buffer
    int read_bytes = read(file_desc, buffer, sizeof(buffer));
    if (read_bytes < 0) {
        // Jika terjadi kesalahan saat membaca
        return 1;
    }
    
    // Menulis isi buffer ke stdout (layar)
    write(1, buffer, read_bytes);  // 1 adalah file descriptor untuk stdout

    // Menutup file setelah selesai
    close(file_desc);

    return 0;
}
```

### Penjelasan Contoh:
- **open()**: Membuka file `example.txt` dan mengembalikan file descriptor yang digunakan untuk merujuk ke file. Jika gagal, mengembalikan nilai negatif.
- **read()**: Membaca isi file ke dalam buffer dan mengembalikan jumlah byte yang dibaca.
- **write()**: Menulis isi buffer ke **stdout** (layar) menggunakan file descriptor `1`, yang merujuk ke **standard output**.
- **close()**: Menutup file setelah selesai.

### Proses System Call
Ketika `read()` atau `write()` dipanggil:
1. Program memasuki **user mode**.
2. Permintaan `read()` menginstruksikan OS untuk mengakses file di memori atau disk, sehingga CPU beralih ke **kernel mode**.
3. Kernel mengeksekusi instruksi `read()` dan mengambil data dari disk.
4. Setelah selesai, hasilnya dikirim ke aplikasi, dan CPU kembali ke **user mode**.

### Contoh System Call dalam Assembly
Jika kita melihat di level assembly, system call pada Linux (untuk arsitektur x86) biasanya menggunakan **interrupt** untuk beralih ke mode kernel. Berikut contoh kode assembly untuk menggunakan system call `write()`:

```assembly
section .data
    msg db 'Hello, world!', 0xA     ; Pesan yang akan ditulis
    len equ $ - msg                 ; Panjang pesan

section .text
    global _start

_start:
    ; System call: write(int fd, const void *buf, size_t count)
    mov eax, 4        ; syscall number untuk write
    mov ebx, 1        ; file descriptor 1 (stdout)
    mov ecx, msg      ; alamat pesan
    mov edx, len      ; panjang pesan
    int 0x80          ; interrupt untuk memanggil kernel

    ; System call: exit(int status)
    mov eax, 1        ; syscall number untuk exit
    xor ebx, ebx      ; status 0
    int 0x80          ; interrupt untuk memanggil kernel
```

### Penjelasan:
- **eax**: Menyimpan nomor system call (`4` untuk `write()` dan `1` untuk `exit()`).
- **ebx**: Menyimpan file descriptor (dalam hal ini, `1` untuk `stdout`).
- **ecx**: Menunjuk ke buffer yang berisi data (pesan).
- **edx**: Menyimpan panjang data yang akan ditulis.
- **int 0x80**: Memanggil interrupt untuk beralih dari user mode ke kernel mode dan mengeksekusi system call.

### Kesimpulan
System call adalah antarmuka penting yang memungkinkan aplikasi berkomunikasi dengan kernel dan perangkat keras dengan aman. Mereka menyediakan fungsi dasar untuk operasi file, proses, jaringan, dan memori yang menjadi fondasi dari sebagian besar operasi sistem operasi.