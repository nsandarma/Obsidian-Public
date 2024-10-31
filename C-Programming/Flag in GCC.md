Dalam GCC (GNU Compiler Collection), terdapat berbagai flag yang dapat digunakan untuk mengontrol bagaimana kompilator bekerja. Berikut ini adalah beberapa flag yang umum digunakan di GCC:

### 1. **Flag Kompilasi Dasar**
- **`-o <file>`**: Menentukan nama file output.
  - Contoh: `gcc -o outputfile sourcefile.c` akan menghasilkan executable dengan nama `outputfile`.
  
- **`-c`**: Mengompilasi file sumber menjadi file object (.o) tanpa melakukan linking.
  - Contoh: `gcc -c sourcefile.c` menghasilkan `sourcefile.o`.

- **`-S`**: Menghasilkan file assembly (.s) dari kode C tanpa melakukan kompilasi menjadi file object.
  - Contoh: `gcc -S sourcefile.c` akan menghasilkan file `sourcefile.s`.

- **`-E`**: Menjalankan preprocessor saja dan tidak melakukan kompilasi lebih lanjut.
  - Contoh: `gcc -E sourcefile.c` akan menghasilkan kode sumber setelah preprocessing.

- **`-g`**: Menambahkan informasi debug ke dalam file output.
  - Contoh: `gcc -g sourcefile.c` memungkinkan executable untuk digunakan dengan debugger seperti `gdb`.

- **`-I <path>`**: Menentukan direktori tambahan untuk mencari header file saat preprocessing.
  - Contoh: `gcc -I /usr/local/include sourcefile.c`.

### 2. **Flag Optimisasi**
- **`-O0`**: Tidak melakukan optimisasi (default).
  - Cocok untuk debugging karena kode dihasilkan lebih mendekati kode sumber aslinya.
  
- **`-O1`**: Optimisasi level dasar. Melakukan optimisasi sederhana yang tidak terlalu memperlambat proses kompilasi.
  
- **`-O2`**: Optimisasi tingkat lanjut. Lebih agresif dibanding `-O1` dengan mengoptimalkan performa dan ukuran file.
  
- **`-O3`**: Optimisasi tingkat tinggi. Melakukan semua optimisasi di level `-O2` ditambah optimisasi yang lebih mahal dan kompleks.

- **`-Os`**: Mengoptimalkan ukuran file hasil kompilasi dengan mengurangi besar file executable.

- **`-Ofast`**: Mengaktifkan optimisasi `-O3` serta beberapa optimisasi yang mungkin melanggar standar kompilasi (misalnya, optimisasi floating-point yang tidak mengikuti standar IEEE).

### 3. **Flag Linker**
- **`-L <path>`**: Menentukan direktori tambahan untuk mencari library saat linking.
  - Contoh: `gcc -L /usr/local/lib -o outputfile sourcefile.c`.

- **`-l <library>`**: Menautkan library eksternal tertentu saat linking.
  - Contoh: `gcc sourcefile.c -lm` akan menautkan library matematika (`libm.so`).
  
- **`-static`**: Memaksa kompilator untuk menggunakan linking statis alih-alih linking dinamis.

- **`-shared`**: Menghasilkan file shared library yang dapat digunakan oleh program lain pada saat runtime.
  - Contoh: `gcc -shared -o libexample.so sourcefile.c` menghasilkan shared library `libexample.so`.

### 4. **Flag Arsitektur dan Platform**
- **`-m32`**: Mengompilasi kode untuk arsitektur 32-bit.
  - Contoh: `gcc -m32 sourcefile.c` menghasilkan executable 32-bit.

- **`-m64`**: Mengompilasi kode untuk arsitektur 64-bit (default pada sistem 64-bit).

### 5. **Flag Warnings dan Errors**
- **`-Wall`**: Mengaktifkan semua peringatan kompilasi umum.
  - Contoh: `gcc -Wall sourcefile.c` akan menampilkan berbagai peringatan umum.

- **`-Werror`**: Mengubah semua peringatan menjadi error.
  - Contoh: `gcc -Werror sourcefile.c` akan menghentikan kompilasi jika ada peringatan.

- **`-Wextra`**: Menambahkan peringatan tambahan selain yang disertakan dengan `-Wall`.
  
- **`-pedantic`**: Memaksa GCC untuk sepenuhnya mengikuti standar bahasa C dan menampilkan peringatan jika ada fitur non-standar.

### 6. **Flag Profiling dan Debugging**
- **`-pg`**: Menambahkan informasi profiling untuk alat seperti `gprof`, yang memungkinkan analisis performa program.
  - Contoh: `gcc -pg sourcefile.c` menghasilkan executable dengan informasi profil untuk analisis performa.

- **`-fsanitize=<type>`**: Mengaktifkan alat pemantauan runtime untuk menemukan bug seperti buffer overflows, undefined behavior, dll.
  - Contoh: `gcc -fsanitize=address sourcefile.c` mengaktifkan AddressSanitizer untuk mendeteksi bug yang berkaitan dengan akses memori yang salah.

### 7. **Flag Multithreading dan Parallelization**
- **`-fopenmp`**: Mengaktifkan dukungan untuk OpenMP, sebuah API untuk pemrograman paralel di lingkungan multi-core.
  - Contoh: `gcc -fopenmp sourcefile.c` memungkinkan program memanfaatkan threading paralel dengan OpenMP.

Dengan menggunakan flag ini, Anda dapat mengontrol secara mendetail bagaimana GCC menangani proses kompilasi, optimisasi, dan linking sesuai kebutuhan proyek Anda.