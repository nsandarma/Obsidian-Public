Dalam C, pustaka (library) dapat dibagi menjadi dua jenis utama: **pustaka statis (static library)** dan **pustaka dinamis (dynamic library)**. Keduanya digunakan untuk menyimpan kode yang dapat digunakan ulang oleh berbagai program, tetapi ada beberapa perbedaan utama dalam cara kerja dan penggunaannya.

### 1. **Pustaka Statis (Static Library)**
   - **Ekstensi file**: Biasanya memiliki ekstensi `.a` di Linux/Unix dan `.lib` di Windows.
   - **Linking**: Pustaka statis digabungkan ke dalam program saat tahap **linking** (kompilasi). Ini berarti bahwa semua kode dari pustaka yang diperlukan oleh program akan disalin ke dalam **file executable** akhir.
   - **Distribusi**: Setelah executable dibuat, pustaka statis tidak perlu didistribusikan bersama program karena sudah menjadi bagian dari executable.
   - **Keuntungan**:
     - Tidak memerlukan pustaka terpisah saat program dijalankan.
     - Tidak ada overhead waktu pemuatan pustaka saat program berjalan, karena sudah menjadi bagian dari program.
     - Program lebih mudah didistribusikan karena hanya bergantung pada file executable.
   - **Kekurangan**:
     - **Ukuran file executable** menjadi lebih besar, karena kode pustaka statis digabungkan ke dalam executable.
     - Jika pustaka diperbarui (misalnya, untuk memperbaiki bug), program perlu **dikompilasi ulang** untuk mengambil perubahan tersebut.

### 2. **Pustaka Dinamis (Dynamic Library)**
   - **Ekstensi file**: Biasanya memiliki ekstensi `.so` (Shared Object) di Linux/Unix dan `.dll` (Dynamic Link Library) di Windows.
   - **Linking**: Pustaka dinamis tidak digabungkan ke dalam executable selama kompilasi. Sebaliknya, program akan melakukan **dynamic linking** saat runtime, di mana pustaka dinamis dimuat ke memori ketika program dijalankan.
   - **Distribusi**: Pustaka dinamis harus disertakan dengan program, atau tersedia di sistem target, agar program dapat berjalan.
   - **Keuntungan**:
     - **Ukuran executable** lebih kecil karena pustaka tidak dimasukkan ke dalam program saat kompilasi.
     - **Pembagian pustaka**: Beberapa program dapat berbagi pustaka yang sama, sehingga menghemat penggunaan memori dan ruang disk.
     - Jika pustaka diperbarui, program tidak perlu dikompilasi ulang, karena pustaka yang diperbarui akan digunakan secara otomatis saat runtime.
   - **Kekurangan**:
     - Pustaka dinamis harus tersedia di sistem saat program dijalankan, jadi jika pustaka hilang atau versinya tidak sesuai, program mungkin gagal dijalankan (missing shared library errors).
     - Sedikit overhead selama runtime untuk memuat pustaka ke dalam memori.

### Contoh Penggunaan:
- **Pustaka Statis**: Cocok untuk program yang perlu berdiri sendiri dan tidak ingin bergantung pada pustaka eksternal selama runtime. Ini sering digunakan dalam perangkat lunak yang didistribusikan tanpa banyak ketergantungan eksternal.
- **Pustaka Dinamis**: Sangat umum di sistem operasi modern seperti Linux dan Windows, di mana pustaka yang sama dapat digunakan oleh beberapa program secara bersamaan. Contoh: banyak program di Linux menggunakan pustaka dinamis seperti `glibc`.

### Ringkasan Perbedaan

| Fitur                   | Pustaka Statis                           | Pustaka Dinamis                             |
|-------------------------|------------------------------------------|---------------------------------------------|
| **Ekstensi File**        | `.a` (Linux), `.lib` (Windows)           | `.so` (Linux), `.dll` (Windows)             |
| **Linking**              | Saat kompilasi (compile-time)            | Saat program berjalan (runtime)             |
| **Ukuran Executable**    | Lebih besar                             | Lebih kecil                                 |
| **Distribusi**           | Pustaka tidak diperlukan saat runtime    | Pustaka harus tersedia di sistem saat runtime |
| **Kinerja Runtime**      | Tidak ada overhead untuk memuat pustaka  | Sedikit overhead untuk memuat pustaka       |
| **Pembaruan Pustaka**    | Perlu dikompilasi ulang untuk pembaruan  | Tidak perlu kompilasi ulang                 |

Dengan memilih antara pustaka statis atau dinamis, Anda dapat menyesuaikan distribusi dan performa program sesuai kebutuhan proyek.

Berikut adalah contoh pembuatan **pustaka statis** dan **pustaka dinamis** sederhana di C.
### Contoh Pustaka Statis

#### 1. Buat File Sumber (misalnya, `mylib.c`)
Ini adalah kode sederhana untuk pustaka:

```c
// mylib.c
#include <stdio.h>

void hello() {
    printf("Hello from static library!\n");
}
```

#### 2. Buat Header File (misalnya, `mylib.h`)
Header file ini digunakan untuk mendeklarasikan fungsi pustaka.

```c
// mylib.h
#ifndef MYLIB_H
#define MYLIB_H

void hello();

#endif
```

#### 3. Kompilasi Pustaka Statis
Kompilasi kode C di atas menjadi pustaka statis dengan perintah berikut:

```bash
gcc -c mylib.c -o mylib.o   # Compile file objek
ar rcs libmylib.a mylib.o   # Buat pustaka statis (libmylib.a)
```

#### 4. Gunakan Pustaka Statis
Sekarang, buat program yang menggunakan pustaka ini.

##### Buat file `main.c`:

```c
// main.c
#include "mylib.h"

int main() {
    hello();  // Panggil fungsi dari pustaka statis
    return 0;
}
```

##### Kompilasi dengan pustaka statis:

```bash
gcc main.c -L. -lmylib -o main   # -L. untuk direktori pustaka, -lmylib untuk pustaka statis
```

Kemudian jalankan program:

```bash
./main
```

Output:

```
Hello from static library!
```

### Contoh Pustaka Dinamis

#### 1. Buat File Sumber (misalnya, `mylib.c`)
Kita akan menggunakan file sumber yang sama dengan contoh pustaka statis.

```c
// mylib.c
#include <stdio.h>

void hello() {
    printf("Hello from dynamic library!\n");
}
```

#### 2. Kompilasi Pustaka Dinamis
Gunakan perintah berikut untuk mengompilasi kode menjadi pustaka dinamis:

```bash
gcc -fPIC -c mylib.c -o mylib.o  # Compile file objek dengan posisi independen (fPIC)
gcc -shared -o libmylib.so mylib.o  # Buat pustaka dinamis (libmylib.so)
```

#### 3. Gunakan Pustaka Dinamis
Sekarang, buat program yang menggunakan pustaka dinamis.

##### Buat file `main.c`:

```c
// main.c
#include "mylib.h"

int main() {
    hello();  // Panggil fungsi dari pustaka dinamis
    return 0;
}
```

##### Kompilasi dengan pustaka dinamis:

```bash
gcc main.c -L. -lmylib -o main   # Kompilasi dengan pustaka dinamis
```

atau menggunakan cara berikut :

```bash
gcc main.c -L. -lmylib -Wl,-rpath=. -o main
```

**Catatan**: Pastikan bahwa pustaka dinamis (`libmylib.so`) dapat ditemukan oleh program pada saat runtime. Misalnya, Anda dapat menambahkan direktori saat ini ke variabel lingkungan `LD_LIBRARY_PATH` di Linux:

```bash
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
```

Kemudian jalankan program:

```bash
./main
```

Output:

```
Hello from dynamic library!
```

### Ringkasan
- **Pustaka statis**: Seluruh kode pustaka disertakan ke dalam file executable saat kompilasi.
- **Pustaka dinamis**: Pustaka dimuat secara dinamis saat runtime dan tidak disertakan dalam executable. pastikan lokasi pustaka dapat dijangkau oleh executable

Dengan contoh ini, Anda bisa melihat perbedaan praktis antara pustaka statis dan dinamis di C.