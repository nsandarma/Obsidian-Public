CUDA (Compute Unified Device Architecture) adalah sebuah platform dan model pemrograman yang dikembangkan oleh NVIDIA, yang memungkinkan para developer untuk memanfaatkan kekuatan GPU (Graphics Processing Unit) untuk melakukan komputasi umum (General-Purpose Computing on GPUs atau GPGPU). Dengan CUDA, pengembang dapat menggunakan bahasa pemrograman seperti C, C++, atau Python untuk menjalankan kode di GPU, bukan hanya di CPU.

Berikut beberapa konsep utama yang perlu dipahami dalam pemrograman CUDA:

### 1. **GPU vs. CPU**
   - **CPU (Central Processing Unit)**: Didesain untuk menangani beberapa thread besar secara cepat, tetapi jumlah core terbatas (biasanya 4-16 core).
   - **GPU**: Memiliki ribuan core kecil yang bekerja secara paralel. GPU sangat cocok untuk masalah komputasi yang dapat dipecah menjadi tugas-tugas kecil yang dapat dilakukan secara bersamaan.

### 2. **Kernel**
   - Kernel adalah fungsi yang dieksekusi di GPU. Ini merupakan unit kerja yang dijalankan oleh banyak thread secara paralel pada GPU.
   - Setiap thread menjalankan kernel yang sama, tetapi biasanya pada data yang berbeda (disebut sebagai "Single Instruction Multiple Threads" atau SIMT).

### 3. **Grid dan Block**
   - **Grid**: Kumpulan dari blocks. Grid digunakan untuk mengorganisasikan parallelism pada tingkat yang lebih tinggi.
   - **Block**: Kumpulan threads yang dieksekusi bersama-sama dalam satu warp di GPU.
   - **Threads**: Unit eksekusi terkecil yang menjalankan kernel pada GPU.

   Setiap block dapat memiliki sejumlah threads yang bekerja secara paralel, dan grid dapat memiliki beberapa block. Secara hierarkis:
   - Grid → Blocks → Threads.

### 4. **Memori pada GPU**
   Ada beberapa tipe memori di CUDA:
   - **Global Memory**: Memori utama GPU, dapat diakses oleh semua threads, tetapi memiliki latensi tinggi.
   - **Shared Memory**: Memori yang lebih cepat dan digunakan bersama oleh threads dalam satu block. Latensinya lebih rendah dibanding global memory.
   - **Local Memory**: Setiap thread memiliki memori sendiri untuk menyimpan variabel yang hanya bisa diakses oleh thread tersebut.
   - **Register**: Tempat penyimpanan yang paling cepat di dalam GPU, diakses oleh setiap thread.
   - **Constant dan Texture Memory**: Jenis memori khusus yang optimal untuk data yang sering dibaca tetapi jarang diubah.

### 5. **Parallelism**
   CUDA menggunakan eksekusi paralel yang dipecah menjadi dua bentuk:
   - **Data Parallelism**: Eksekusi kernel pada banyak elemen data secara bersamaan.
   - **Task Parallelism**: Memungkinkan beberapa tugas yang berbeda untuk dieksekusi secara paralel.

### 6. **Program CUDA**
   Struktur dasar dari program CUDA biasanya terdiri dari bagian berikut:
   - **Memindahkan data dari CPU (host) ke GPU (device)**.
   - **Menjalankan kernel pada GPU**.
   - **Memindahkan hasil dari GPU kembali ke CPU**.

Contoh sederhana kernel CUDA (C++):

```cpp
__global__ void add(int *a, int *b, int *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    int a[5], b[5], c[5];
    int *dev_a, *dev_b, *dev_c;

    // Allocate memory on the GPU
    cudaMalloc((void**)&dev_a, 5 * sizeof(int));
    cudaMalloc((void**)&dev_b, 5 * sizeof(int));
    cudaMalloc((void**)&dev_c, 5 * sizeof(int));

    // Copy arrays from host (CPU) to device (GPU)
    cudaMemcpy(dev_a, a, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, 5 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 5 threads
    add<<<1, 5>>>(dev_a, dev_b, dev_c);

    // Copy result back to host (CPU)
    cudaMemcpy(c, dev_c, 5 * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the memory allocated on GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
```

### 7. **Optimasi**
   - **Coalesced Memory Access**: Agar akses memori lebih efisien, threads harus mengakses memori global dalam pola yang teratur.
   - **Memory Bank Conflicts**: Shared memory diorganisasikan dalam "banks", dan akses simultan oleh threads pada bank yang sama menyebabkan konflik dan memperlambat eksekusi.
   - **Occupancy**: Rasio antara jumlah threads aktif dengan jumlah maksimum threads yang dapat dijalankan pada GPU. Meningkatkan occupancy dapat meningkatkan efisiensi eksekusi kernel.

### 8. **Alat dan Pustaka**
   - **Thrust**: Library parallel high-level untuk CUDA, mirip dengan C++ Standard Template Library (STL).
   - **cuBLAS dan cuDNN**: Library untuk komputasi numerik yang memanfaatkan GPU, seperti operasi matriks dan deep learning.

Dengan menggunakan CUDA, aplikasi yang melibatkan komputasi intensif seperti machine learning, grafis, dan simulasi fisika dapat diakselerasi secara signifikan.

# Perbedaan NVCC dan NVRTC

`nvcc` (NVIDIA CUDA Compiler) dan `nvrtc` (NVIDIA Runtime Compilation) adalah dua compiler berbeda yang digunakan untuk mengompilasi kode CUDA, tetapi memiliki peran dan pendekatan yang berbeda dalam proses kompilasi:

### 1. `nvcc` (NVIDIA CUDA Compiler)

- **Jenis Kompilasi**: Kompilasi offline (Ahead-of-Time Compilation).
- **Proses Kompilasi**: `nvcc` adalah compiler utama untuk CUDA yang digunakan untuk mengompilasi kode CUDA sebelum eksekusi program. Ia mengompilasi file `.cu` menjadi kode objek atau executable yang dapat dijalankan secara langsung.
- **Output**: Menghasilkan file biner (kode objek atau executable) yang siap dijalankan, biasanya dalam bentuk `.ptx`, `.cubin`, atau biner executable.
- **Penggunaan**: `nvcc` sering digunakan dalam pengembangan aplikasi GPU yang memiliki arsitektur dan kode yang sudah tetap dan tidak memerlukan perubahan dinamis saat runtime.
- **Kelebihan**:
  - Lebih efisien karena semua kompilasi selesai sebelum eksekusi program.
  - Mendukung seluruh fitur CUDA dan pustaka eksternal karena kode selesai di-compile sebelum runtime.
- **Kekurangan**:
  - Tidak fleksibel untuk kompilasi dinamis, sehingga kurang ideal jika kode perlu dimodifikasi atau di-generate pada runtime.

### 2. `nvrtc` (NVIDIA Runtime Compilation)

- **Jenis Kompilasi**: Kompilasi pada saat runtime (Just-in-Time Compilation).
- **Proses Kompilasi**: `nvrtc` adalah compiler CUDA yang digunakan untuk mengompilasi kode CUDA secara dinamis saat runtime. Alih-alih file `.cu`, ia menggunakan string kode sumber yang dihasilkan atau dimodifikasi pada runtime.
- **Output**: Menghasilkan kode PTX (Parallel Thread Execution) atau objek lain yang dapat dimuat dan dijalankan secara langsung oleh aplikasi pada runtime.
- **Penggunaan**: `nvrtc` sangat berguna dalam aplikasi yang membutuhkan fleksibilitas untuk membuat atau memodifikasi kode CUDA secara dinamis saat runtime, misalnya untuk optimasi runtime atau aplikasi yang sangat bergantung pada input.
- **Kelebihan**:
  - Memungkinkan kompilasi dan eksekusi kode CUDA yang dihasilkan secara dinamis selama runtime.
  - Ideal untuk aplikasi yang perlu menyesuaikan kernel CUDA mereka secara dinamis.
- **Kekurangan**:
  - Lebih lambat daripada kompilasi statis karena kompilasi dilakukan saat runtime.
  - Mendukung subset dari fitur `nvcc` dan tidak kompatibel dengan semua pustaka eksternal.

### Kapan Menggunakan `nvcc` vs `nvrtc`

- **Gunakan `nvcc`** jika Anda memiliki kernel CUDA yang stabil dan tidak perlu dimodifikasi pada runtime, dan Anda ingin performa eksekusi yang optimal.
- **Gunakan `nvrtc`** jika Anda perlu fleksibilitas dalam mendefinisikan atau mengoptimalkan kernel CUDA berdasarkan kondisi runtime atau masukan pengguna, atau jika Anda perlu men-generate kernel CUDA secara dinamis.