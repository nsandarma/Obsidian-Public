Assembly **x86** adalah bahasa pemrograman tingkat rendah yang digunakan untuk menginstruksikan prosesor x86 untuk menjalankan tugas tertentu. Bahasa assembly memberikan instruksi langsung ke prosesor, dan instruksi-instruksi ini mencerminkan apa yang benar-benar dilakukan oleh hardware.

Untuk lebih memahami **assembly x86**, kita akan membahas beberapa konsep penting, termasuk register, instruksi dasar, mode pengalamatan, dan sistem operasi. 

### 1. **Register pada x86**
Register adalah tempat penyimpanan data yang paling cepat di dalam CPU. Pada arsitektur **x86 (32-bit)**, terdapat beberapa **General-Purpose Registers (GPR)** yang bisa digunakan untuk menyimpan data selama eksekusi instruksi.

Beberapa register utama pada **x86-32** meliputi:
- **EAX**: Akumulator untuk operasi aritmatika dan logika. Sering digunakan untuk menyimpan hasil operasi.
- **EBX**: Biasanya digunakan sebagai base pointer untuk pengalamatan memori.
- **ECX**: Register penghitung, sering digunakan untuk loop dan operasi pengulangan.
- **EDX**: Data register, digunakan untuk menyimpan nilai yang lebih besar dalam operasi pembagian dan perkalian.
- **ESI/EDI**: Source Index dan Destination Index, digunakan untuk operasi terkait string.
- **ESP**: Stack Pointer, menunjuk ke posisi teratas di stack.
- **EBP**: Base Pointer, digunakan untuk menyimpan basis dari stack frame selama pemanggilan fungsi.

Dalam mode **x86-64 (64-bit)**, register diperluas menjadi **RAX**, **RBX**, **RCX**, dan seterusnya, dengan register tambahan seperti **R8-R15**.

### 2. **Instruksi Dasar Assembly x86**
Instruksi di assembly x86 adalah perintah yang diberikan ke CPU untuk melakukan operasi tertentu. Beberapa instruksi dasar meliputi:

#### a. **MOV** (Move)
Memindahkan data dari satu tempat ke tempat lain.
```assembly
mov eax, 5     ; Memindahkan nilai 5 ke register EAX
mov ebx, eax   ; Memindahkan nilai dalam EAX ke EBX
```

#### b. **ADD / SUB** (Penjumlahan / Pengurangan)
Menambahkan atau mengurangi nilai pada register.
```assembly
add eax, 3     ; Menambahkan 3 ke nilai dalam EAX (EAX = EAX + 3)
sub eax, 1     ; Mengurangi 1 dari nilai dalam EAX (EAX = EAX - 1)
```

#### c. **MUL / DIV** (Perkalian / Pembagian)
Digunakan untuk operasi aritmatika yang lebih kompleks.
```assembly
mul ebx        ; Mengalikan nilai dalam EAX dengan EBX (hasil di EAX)
div ebx        ; Membagi EAX dengan EBX (hasil di EAX, sisa di EDX)
```

#### d. **CMP** (Compare)
Membandingkan dua nilai dan mengatur flag (digunakan untuk percabangan).
```assembly
cmp eax, 10    ; Membandingkan EAX dengan 10
```

#### e. **JMP / JZ / JNZ** (Jump)
Instruksi percabangan untuk melompat ke bagian lain kode.
```assembly
jmp label      ; Melompat ke label tertentu
jz equal_label ; Lompat jika hasil perbandingan adalah nol (equal)
jnz not_equal_label ; Lompat jika tidak sama (not equal)
```

#### f. **CALL / RET** (Memanggil Fungsi)
`CALL` digunakan untuk memanggil subrutin (fungsi) dan `RET` digunakan untuk kembali dari subrutin.
```assembly
call my_function   ; Memanggil fungsi
ret                ; Kembali dari fungsi
```

### 3. **Mode Pengalamatan (Addressing Modes)**
Mode pengalamatan adalah cara CPU mengakses data dari memori. Dalam assembly x86, ada beberapa jenis mode pengalamatan:

- **Immediate Addressing**: Nilai langsung ada dalam instruksi.
  ```assembly
  mov eax, 10  ; Memindahkan nilai 10 langsung ke EAX
  ```

- **Register Addressing**: Data diambil dari register.
  ```assembly
  mov eax, ebx ; Memindahkan nilai EBX ke EAX
  ```

- **Direct Addressing**: Mengakses memori dengan alamat langsung.
  ```assembly
  mov eax, [0x1234] ; Memindahkan data di alamat 0x1234 ke EAX
  ```

- **Indirect Addressing**: Mengakses memori melalui register yang berisi alamat.
  ```assembly
  mov eax, [ebx]    ; Memindahkan data dari alamat yang ditunjuk EBX ke EAX
  ```

- **Indexed Addressing**: Menggunakan register dan offset.
  ```assembly
  mov eax, [ebx+4]  ; Mengambil data dari alamat EBX + 4
  ```

### 4. **Sistem Operasi dan Syscall**
Pada arsitektur x86, terutama dalam konteks Linux, instruksi `syscall` (atau `int 0x80` pada x86-32) digunakan untuk berinteraksi dengan sistem operasi untuk melakukan operasi seperti membaca file, menulis ke layar, atau keluar dari program.

Contoh syscall untuk **exit** di **x86-64**:
```assembly
mov rax, 60    ; Syscall untuk exit di x86-64
mov rdi, 0     ; Kode keluar (0 berarti normal)
syscall        ; Panggil sistem operasi
```

Pada x86-32, hal ini dilakukan menggunakan interrupt `int 0x80`:
```assembly
mov eax, 1     ; Syscall untuk exit di x86-32
mov ebx, 0     ; Kode keluar (0 berarti normal)
int 0x80       ; Panggil sistem operasi
```

### 5. **Stack di Assembly x86**
Stack adalah struktur data LIFO (Last In, First Out) yang digunakan untuk menyimpan data sementara, seperti parameter fungsi dan alamat pengembalian. Beberapa instruksi yang sering digunakan dengan stack:
- **PUSH**: Menyimpan data ke stack.
- **POP**: Mengambil data dari stack.

Contoh:
```assembly
push eax       ; Menyimpan nilai EAX ke stack
pop ebx        ; Mengambil nilai dari stack dan memasukkannya ke EBX
```

Stack sangat penting dalam pemanggilan fungsi dan pengelolaan alokasi memori sementara.

### 6. **Contoh Program Sederhana**
Program sederhana untuk menambahkan dua angka dan keluar.

```assembly
section .data      ; Bagian data (opsional, untuk data statis)

section .text      ; Bagian kode (program utama)
    global _start  ; Titik awal program

_start:
    mov eax, 5     ; Memindahkan nilai 5 ke register EAX
    add eax, 3     ; Menambahkan 3 ke EAX (EAX = 5 + 3)

    ; Keluar dari program
    mov eax, 1     ; Syscall exit untuk x86-32
    mov ebx, 0     ; Kode keluar (0 untuk sukses)
    int 0x80       ; Panggil syscall exit
```

### Kesimpulan
Assembly **x86** memberikan kontrol langsung atas hardware, memungkinkan programmer mengakses register, memori, dan instruksi CPU secara detail. Meskipun sangat kuat, assembly lebih kompleks daripada bahasa pemrograman tingkat tinggi, dan umumnya digunakan dalam situasi yang memerlukan performa tinggi atau pengelolaan resource yang sangat spesifik. 

### Eksekusi Program
Program di atas dapat ditulis dalam file dengan ekstensi `.asm` dan dirakit menggunakan **NASM** pada sistem Linux.

Langkah-langkah eksekusinya:
1. **Assemble** program menggunakan NASM:
   ```
   nasm -f elf32 sum.asm -o sum.o
   ```
2. **Link** file objek yang dihasilkan:
   ```
   ld -m elf_i386 -o sum sum.o
   ```
3. **Jalankan** program:
   ```
   ./sum
   ```

Pada akhir program, hasil penjumlahan akan tersimpan di alamat memori yang ditentukan oleh variabel `result`.


1 byte = 8 bit
1 byte = 2 hex