Proses kompilasi program C terdiri dari beberapa langkah utama, yang semuanya terjadi secara bertahap untuk menghasilkan file executable yang bisa dijalankan oleh sistem. Berikut adalah penjelasan detail dari proses kompilasi program C:

### 1. **Preprocessing (Pra-pemrosesan)**
   Pada tahap ini, file sumber C (.c) diproses oleh **preprocessor**. Preprocessor bertugas untuk memproses semua perintah yang diawali dengan tanda `#`, seperti `#include`, `#define`, dan `#ifdef`. Berikut beberapa langkah yang terjadi:

   - **Penggantian Directive `#include`**: File header yang disertakan menggunakan `#include` akan digabungkan ke dalam file sumber.
   - **Makro Didefinisikan**: Semua makro yang didefinisikan dengan `#define` akan digantikan oleh nilai atau ekspresi yang didefinisikan di tempat yang digunakan dalam kode.
   - **Mengabaikan Komentar**: Komentar dalam kode akan dihapus.
   - **Pengondisian Kompilasi**: Direktif seperti `#ifdef` dan `#endif` akan diproses, yaitu bagian kode tertentu dapat dimasukkan atau diabaikan berdasarkan kondisi yang diberikan.

   Hasil dari tahap ini adalah file *intermediate* yang sudah tidak mengandung direktif pra-pemrosesan. File ini biasanya berakhiran `.i` atau `.ii` (untuk C++).

   ```bash
   gcc -E program.c -o program.i
   ```

### 2. **Compilation (Kompilasi)**
   Pada tahap ini, kode sumber yang telah dipra-pemrosesan akan dikompilasi oleh **compiler** menjadi kode **assembly**. Kode assembly adalah representasi low-level dari kode yang lebih dekat ke instruksi mesin tetapi masih dalam format yang dapat dibaca oleh manusia.

   Kode assembly yang dihasilkan biasanya berakhiran `.s`. Kompiler bertugas untuk menganalisis kode dan melakukan optimasi untuk menghasilkan kode yang lebih efisien.

   ```bash
   gcc -S program.i -o program.s
   ```

### 3. **Assembly (Assembling)**
   Tahap ini adalah proses mengubah kode assembly menjadi **kode mesin**. Ini dilakukan oleh **assembler** yang menerjemahkan instruksi assembly ke dalam instruksi biner yang bisa dipahami oleh prosesor.

   Hasil dari tahap ini adalah file objek berakhiran `.o` atau `.obj`. File ini berisi kode biner yang masih terpisah dari file objek lain yang mungkin diperlukan.

   ```bash
   gcc -c program.s -o program.o
   ```

### 4. **Linking (Penggabungan)**
   Tahap terakhir adalah **linking**, di mana **linker** menggabungkan file objek yang telah dihasilkan bersama dengan library yang diperlukan (misalnya, pustaka standar C) menjadi sebuah **executable**. Pada tahap ini, semua referensi ke fungsi dan variabel yang dideklarasikan secara eksternal (seperti fungsi di pustaka standar C `printf`, `malloc`, dll.) diselesaikan dan digabungkan.

   File yang dihasilkan dari tahap ini adalah file executable yang dapat dijalankan (misalnya, `program.exe` pada Windows atau `./program` pada Linux).

   ```bash
   gcc program.o -o program
   ```

### Ilustrasi Ringkas:
Misalkan kita memiliki file `main.c` yang sederhana:
```c
#include <stdio.h>

int main() {
    printf("Hello, world!\n");
    return 0;
}
```

Prosesnya secara singkat:
1. **Preprocessing**:
   - File header `stdio.h` akan disertakan.
   - Komentar dan makro dihapus.

2. **Compilation**:
   - Kode C diterjemahkan menjadi kode assembly.

3. **Assembly**:
   - Kode assembly diterjemahkan menjadi kode mesin.

4. **Linking**:
   - Kode mesin dari file objek digabungkan dengan pustaka standar untuk menghasilkan file yang dapat dieksekusi.

### **Proses Kompilasi Secara Otomatis dengan GCC**:
Semua langkah di atas sebenarnya dapat dilakukan dalam satu perintah dengan GCC:

```bash
gcc main.c -o program
```

Perintah ini langsung menggabungkan semua tahapan: preprocessing, compilation, assembly, dan linking.

### Tools yang Terlibat dalam Kompilasi
- **Preprocessor**: Menangani pra-pemrosesan.
- **Compiler**: Mengubah kode sumber C menjadi assembly.
- **Assembler**: Mengubah kode assembly menjadi file objek.
- **Linker**: Menggabungkan file objek menjadi executable.

Proses kompilasi bisa bervariasi tergantung pada opsi yang diberikan ke kompilator dan lingkungan pengembangan yang digunakan.