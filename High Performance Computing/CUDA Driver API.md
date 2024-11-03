Berikut adalah penjelasan rinci tentang fungsi-fungsi yang perlu dilakukan untuk menjalankan kernel CUDA dari Python menggunakan `ctypes`. Setiap langkah mencakup persiapan, pengaturan perangkat, dan eksekusi kernel.

[Cuda Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
### 1. **Memuat Library CUDA**
```python
libcuda = ctypes.cdll.LoadLibrary('libcuda.so')
```
- **Fungsi**: Memuat library driver CUDA (`libcuda.so`) ke dalam program Python. Library ini berisi semua fungsi API yang diperlukan untuk berinteraksi dengan GPU.

### 2. **Menginisialisasi Driver CUDA**
```python
check_cuda_error(libcuda.cuInit(0))
```
- **Fungsi**: Menginisialisasi driver CUDA. Ini adalah langkah pertama sebelum menggunakan fungsi-fungsi CUDA lainnya. Parameter `0` menunjukkan bahwa tidak ada opsi tambahan yang diberikan.

### 3. **Mendapatkan Jumlah Perangkat CUDA**
```python
device_count = ctypes.c_int(0)
check_cuda_error(libcuda.cuDeviceGetCount(ctypes.byref(device_count)))
```
- **Fungsi**: Mengambil jumlah perangkat CUDA yang tersedia di sistem. Hasilnya disimpan dalam variabel `device_count`.

### 4. **Mendapatkan Perangkat CUDA**
```python
device = ctypes.c_void_p()
check_cuda_error(libcuda.cuDeviceGet(ctypes.byref(device), 0))
```
- **Fungsi**: Mengambil referensi ke perangkat CUDA pertama (dalam hal ini, perangkat dengan indeks `0`). Ini memberi tahu program GPU mana yang akan digunakan untuk menjalankan kernel.

### 5. **Membuat Konteks CUDA**
```python
context = ctypes.c_void_p()
check_cuda_error(libcuda.cuCtxCreate(ctypes.byref(context), 0, device))
```
- **Fungsi**: Membuat konteks CUDA untuk perangkat yang telah dipilih. Konteks ini diperlukan agar aplikasi dapat berinteraksi dengan GPU. Parameter `0` menunjukkan bahwa tidak ada opsi khusus untuk konteks yang dibuat.

### 6. **Mempersiapkan Data Input dan Output**
```python
a_np = np.array(A, dtype=np.float32)
b_np = np.array(B, dtype=np.float32)
c_np = np.zeros(N, dtype=np.float32)
```
- **Fungsi**: Menyiapkan data input (`A`, `B`) dan output (`C`). Data ini perlu dalam format yang tepat untuk dioperasikan oleh kernel CUDA.

### 7. **Mengalokasikan Memori di Perangkat**
```python
check_cuda_error(libcuda.cuMemAlloc(ctypes.byref(a_device), a_np.nbytes))
check_cuda_error(libcuda.cuMemAlloc(ctypes.byref(b_device), b_np.nbytes))
check_cuda_error(libcuda.cuMemAlloc(ctypes.byref(c_device), c_np.nbytes))
```
- **Fungsi**: Mengalokasikan memori di GPU untuk data input dan output. Memori ini harus cukup besar untuk menampung data yang akan diproses oleh kernel.

### 8. **Menyalin Data dari Host ke Device**
```python
check_cuda_error(libcuda.cuMemcpyHtoD(a_device, a_np.ctypes.data, a_np.nbytes))
check_cuda_error(libcuda.cuMemcpyHtoD(b_device, b_np.ctypes.data, b_np.nbytes))
```
- **Fungsi**: Menyalin data dari memori host (CPU) ke memori perangkat (GPU). Ini penting karena kernel akan beroperasi pada data yang ada di GPU.

### 9. **Menghitung Ukuran Grid dan Block**
```python
grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
```
- **Fungsi**: Menghitung jumlah blok (grid size) yang diperlukan untuk memproses seluruh data. Ukuran blok biasanya diatur untuk memaksimalkan penggunaan GPU.

### 10. **Menjalankan Kernel**
```python
result = libcuda.cuLaunchKernel(kernel_func, grid_size, 1, 1, BLOCK_SIZE, 1, 1, 0, ctypes.c_void_p(0), kernel_params, ctypes.c_void_p(0))
```
- **Fungsi**: Memanggil kernel CUDA yang telah dikompilasi. Ini adalah langkah yang sebenarnya mengeksekusi kode pada GPU. Parameter termasuk jumlah grid dan blok, ukuran shared memory, dan parameter kernel.

### 11. **Menyalin Data dari Device ke Host**
```python
check_cuda_error(libcuda.cuMemcpyDtoH(c_np.ctypes.data, c_device, c_np.nbytes))
```
- **Fungsi**: Menyalin hasil dari memori perangkat (GPU) kembali ke memori host (CPU) setelah eksekusi kernel selesai.

### 12. **Membebaskan Memori di Device**
```python
libcuda.cuMemFree(a_device)
libcuda.cuMemFree(b_device)
libcuda.cuMemFree(c_device)
```
- **Fungsi**: Membebaskan memori yang dialokasikan di GPU setelah selesai digunakan. Ini penting untuk menghindari kebocoran memori.

### 13. **Menghancurkan Konteks CUDA**
```python
libcuda.cuCtxDestroy(context)
```
- **Fungsi**: Menghancurkan konteks CUDA yang telah dibuat. Ini menyelesaikan interaksi dengan GPU dan membebaskan sumber daya yang terkait dengan konteks tersebut.

### Kesimpulan
Setiap langkah di atas memiliki perannya sendiri dalam mengatur dan menjalankan kernel CUDA. Pastikan untuk melakukan setiap langkah dengan benar untuk memastikan bahwa kernel Anda berjalan dengan baik. Jika ada masalah atau kesalahan yang muncul, pastikan untuk memeriksa kode dan memverifikasi bahwa setiap panggilan API CUDA berhasil. Jika ada pertanyaan lebih lanjut, silakan tanyakan!