Dalam CUDA, *threads*, *grids*, dan *blocks* adalah konsep fundamental untuk menjalankan komputasi paralel pada GPU. Mari kita lihat satu per satu:

### 1. Thread
Sebuah *thread* adalah unit terkecil dari eksekusi dalam CUDA. Setiap *thread* menjalankan kernel CUDA (fungsi yang dijalankan di GPU) dengan instruksi yang sama, tetapi bisa bekerja pada data yang berbeda. Dalam CUDA, ribuan bahkan jutaan *threads* dapat dijalankan secara paralel.

### 2. Block
Sebuah *block* adalah sekumpulan *threads* yang dieksekusi bersama-sama. Setiap *block* memiliki batas maksimum jumlah *threads* (biasanya 1024 *threads* per *block* tergantung arsitektur GPU). *Threads* dalam *block* yang sama dapat berkomunikasi satu sama lain dengan menggunakan memori lokal yang disebut **shared memory**. *Threads* dalam *block* yang sama juga dapat melakukan sinkronisasi (menunggu semua *threads* selesai pada titik tertentu sebelum melanjutkan).

### 3. Grid
*Grid* adalah kumpulan *blocks* yang menjalankan kernel yang sama. Saat kernel diluncurkan, kita menentukan jumlah *blocks* dalam *grid* dan jumlah *threads* dalam setiap *block*. Jadi, ketika kernel dijalankan, *grid* membentuk "jaringan" *blocks* yang masing-masing berisi *threads*. 

Penting untuk memahami bahwa meskipun *threads* dalam *block* yang sama dapat berkomunikasi, *threads* pada *block* yang berbeda tidak bisa, kecuali melalui memori global yang lebih lambat dibanding **shared memory**.

### Contoh Studi Kasus: Penjumlahan Dua Vektor
Misalkan kita ingin menambahkan dua vektor besar, katakanlah berukuran satu juta elemen. Dengan CUDA, kita dapat membagi pekerjaan ini menjadi banyak *threads* yang berjalan secara paralel.

### Langkah-langkah Implementasi:
1. **Tentukan Jumlah Threads per Block** - Misalnya, kita memilih 256 *threads* per *block*.
2. **Hitung Jumlah Blocks yang Dibutuhkan** - Misalnya, jika kita memiliki satu juta elemen, kita memerlukan \( \text{total\_threads} = 1,000,000 \), maka kita akan membagi ini ke dalam beberapa *blocks*, di mana:
   $$
   \text{num\_blocks} = \frac{\text{total\_threads}}{\text{threads\_per\_block}} = \frac{1,000,000}{256} \approx 3907
   $$

Dengan pengaturan ini, kita akan meluncurkan kernel CUDA yang terdiri dari 3907 *blocks*, di mana setiap *block* memiliki 256 *threads*.

### Contoh Kode CUDA
```cpp
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000000;
    float *a, *b, *c;             // host copies
    float *d_a, *d_b, *d_c;       // device copies
    int size = n * sizeof(float);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Setup input values
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch vectorAdd() kernel on GPU with n threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Clean up
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

### Penjelasan Kode
1. **Kernel `vectorAdd`** - Kernel ini mengambil indeks *thread* dalam *grid* yang dihitung dengan `threadIdx.x + blockIdx.x * blockDim.x` untuk mengakses elemen yang sesuai dari vektor.
2. **Peluncuran Kernel** - `vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);` meluncurkan kernel dengan `blocksPerGrid` *blocks* dan `threadsPerBlock` *threads* per *block*. Perhitungan ini memastikan bahwa setiap elemen vektor `a` dan `b` dijumlahkan sesuai dengan indeksnya di `c`.
3. **Sinkronisasi dan Transfer Data** - Hasilnya ditransfer kembali ke CPU dengan `cudaMemcpy`.

### Studi Kasus Lain: Perkalian Matriks
Dalam kasus perkalian matriks besar, kita bisa memanfaatkan *grid* dan *block* untuk mendistribusikan pekerjaan sehingga setiap *thread* bertanggung jawab untuk menghitung satu elemen dari hasil matriks.

Dengan membagi pekerjaan seperti ini, CUDA memungkinkan pemrosesan paralel yang efisien, meningkatkan kinerja aplikasi yang memerlukan komputasi intensif, seperti dalam bidang grafis, simulasi, machine learning, dan data science.