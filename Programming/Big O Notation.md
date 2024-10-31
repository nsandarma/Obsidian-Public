Big O notation adalah cara untuk menggambarkan kompleksitas algoritma, khususnya dalam mengukur efisiensi waktu (time complexity) dan ruang (space complexity) seiring dengan bertambahnya ukuran input (biasanya dilambangkan dengan `n`). Ini digunakan untuk mengevaluasi bagaimana performa suatu algoritma saat skala input menjadi besar.

### Penjelasan:
Big O notation memberikan gambaran tentang batas terburuk (worst-case scenario) dalam hal berapa lama suatu algoritma akan berjalan atau berapa banyak memori yang dibutuhkan seiring dengan pertumbuhan ukuran input.

### Notasi Umum:
1. **O(1)** - **Constant Time**: Waktu eksekusi tidak tergantung pada ukuran input. Contohnya adalah mengakses elemen array dengan indeks tertentu.
   - Contoh: Mengambil elemen ke-3 dari array: `arr[2]`.

2. **O(log n)** - **Logarithmic Time**: Waktu eksekusi tumbuh secara logaritmik. Ini biasanya terjadi dalam algoritma yang membagi masalah menjadi dua bagian pada setiap langkah, seperti dalam binary search.
   - Contoh: Pencarian biner (binary search).

3. **O(n)** - **Linear Time**: Waktu eksekusi tumbuh secara linier dengan ukuran input. Artinya, jika ukuran input bertambah, waktu yang diperlukan juga bertambah secara proporsional.
   - Contoh: Melakukan iterasi pada array.

4. **O(n log n)** - **Linearithmic Time**: Ini biasanya ditemukan pada algoritma pengurutan yang efisien seperti merge sort dan quicksort. Kombinasi linear dan logaritmik.
   - Contoh: Merge Sort, Quick Sort.

5. **O(n²)** - **Quadratic Time**: Waktu eksekusi tumbuh sebanding dengan kuadrat ukuran input. Algoritma dengan kompleksitas ini cenderung lambat untuk ukuran input yang besar.
   - Contoh: Bubble Sort, Insertion Sort.

6. **O(2ⁿ)** - **Exponential Time**: Waktu eksekusi bertambah secara eksponensial dengan ukuran input. Algoritma dengan kompleksitas ini biasanya sangat lambat bahkan untuk ukuran input yang relatif kecil.
   - Contoh: Algoritma untuk masalah kombinatorik seperti brute force pada problem subset sum.

7. **O(n!)** - **Factorial Time**: Waktu eksekusi tumbuh sebanding dengan faktorial dari ukuran input. Ini biasanya ditemukan dalam algoritma brute-force untuk masalah optimasi seperti traveling salesman problem.
   - Contoh: Brute force untuk permutasi semua elemen.

### Contoh:
Jika Anda memiliki algoritma untuk menemukan elemen terbesar dalam array, kompleksitas waktunya akan menjadi:
- **O(n)**, karena Anda perlu memeriksa setiap elemen dalam array.

Namun, jika Anda memiliki algoritma yang mengurutkan array menggunakan bubble sort, kompleksitasnya akan menjadi:
- **O(n²)**, karena untuk setiap elemen, Anda mungkin perlu membandingkannya dengan setiap elemen lainnya.

### Kenapa Big O Penting?
Big O membantu Anda untuk:
- Memilih algoritma yang lebih efisien.
- Memahami batas performa dari algoritma tersebut, terutama untuk ukuran input yang besar.
- Menghindari algoritma yang akan melambat secara eksponensial atau kuadratik saat skala data bertambah.

Ini menjadi alat penting dalam pengembangan perangkat lunak, terutama saat menangani data dalam skala besar atau memerlukan kinerja tinggi.

![[big_o_notation.png]]
Untuk mendapatkan pemahaman yang lebih dalam tentang Big O notation, kita perlu menjelajahi beberapa konsep fundamental dan bagaimana Big O diterapkan dalam berbagai konteks. Mari kita mulai dengan beberapa aspek yang lebih mendetail:

### 1. **Fokus pada Pertumbuhan Skala (Growth Rate)**
Big O notation tidak mengukur waktu eksekusi aktual dari sebuah algoritma. Sebaliknya, ia berfokus pada **laju pertumbuhan** waktu atau ruang yang dibutuhkan seiring bertambahnya ukuran input.

Misalnya, perbedaan antara algoritma dengan kompleksitas waktu **O(n)** dan **O(n²)** bukan sekadar masalah konstanta kecil. Perbedaan ini dapat menjadi signifikan ketika ukuran input bertambah besar.

- Jika Anda punya algoritma dengan **O(n)** dan input sebesar 1000, maka perkiraan waktu eksekusi adalah sebanding dengan 1000 operasi.
- Jika algoritma Anda adalah **O(n²)**, untuk input yang sama, waktu eksekusi akan menjadi sekitar 1.000.000 operasi.

Seiring pertumbuhan input, efek perbedaan ini akan semakin besar. Inilah alasan utama mengapa Big O penting: Ini membantu Anda memperkirakan bagaimana algoritma akan berskala, bukan hanya bagaimana ia bekerja pada input kecil.

### 2. **Mengabaikan Konstanta dan Istilah Kecil**
Big O notation hanya mempertimbangkan **faktor dominan** dari fungsi yang menggambarkan kompleksitas. Kita mengabaikan konstanta dan istilah-istilah yang kurang signifikan dalam pertumbuhan skala.

Misalnya, algoritma dengan kompleksitas waktu:

- **T(n) = 3n² + 5n + 1000**  
  Dalam Big O notation, ini akan disederhanakan menjadi **O(n²)**, karena ketika `n` menjadi sangat besar, kontribusi dari 5n dan 1000 menjadi semakin tidak signifikan dibandingkan dengan `n²`.

Ini berarti Big O membantu kita fokus pada istilah yang paling berpengaruh saat ukuran input tumbuh.

### 3. **Batas Terburuk (Worst-Case) vs. Batas Rata-Rata (Average-Case)**
Big O notation sering kali digunakan untuk menggambarkan **worst-case complexity**, yaitu jumlah maksimum waktu atau ruang yang diperlukan oleh algoritma untuk input dalam skenario terburuk. Ini membantu kita memahami batas maksimum kinerja algoritma ketika menghadapi situasi terburuk.

Namun, ada juga konsep **average-case complexity**, yang menggambarkan kinerja rata-rata algoritma di berbagai input. Beberapa algoritma memiliki perbedaan besar antara worst-case dan average-case. Contoh yang umum adalah **QuickSort**:
- Worst-case complexity dari QuickSort adalah **O(n²)** jika partisi yang dipilih buruk.
- Average-case complexity dari QuickSort adalah **O(n log n)**, yang jauh lebih efisien.

Jadi, memahami perbedaan antara kedua jenis kompleksitas ini sangat penting dalam pemilihan algoritma yang tepat untuk situasi tertentu.

### 4. **Trade-offs antara Waktu dan Ruang (Time-Space Trade-off)**
Algoritma sering kali memiliki **trade-offs** antara penggunaan waktu dan ruang. Sebuah algoritma yang lebih cepat (memiliki kompleksitas waktu lebih baik) mungkin membutuhkan lebih banyak ruang memori (ruang ekstra untuk menyimpan data sementara), dan sebaliknya.

Sebagai contoh:
- Algoritma **merge sort** memiliki kompleksitas waktu **O(n log n)**, yang sangat baik untuk pengurutan, tetapi membutuhkan ruang tambahan sebesar **O(n)** untuk menyimpan array tambahan selama proses merging.
- Algoritma **quick sort** juga memiliki kompleksitas waktu **O(n log n)** pada average case, tetapi membutuhkan ruang tambahan hanya **O(log n)**, yang biasanya lebih kecil.

Saat memilih algoritma, pertimbangan trade-offs antara waktu dan ruang ini penting, terutama dalam aplikasi yang memerlukan optimalisasi keduanya.

### 5. **Perbandingan Berbagai Kompleksitas dengan Contoh Nyata**
Agar lebih jelas, mari lihat bagaimana berbagai Big O complexities berpengaruh pada algoritma untuk input dengan ukuran `n = 1.000.000`:

| Kompleksitas  | Rumus Perkiraan (n=1.000.000) | Contoh Algoritma/Operasi           |
|---------------|-------------------------------|------------------------------------|
| **O(1)**      | 1                             | Mengakses elemen array             |
| **O(log n)**  | ~20                           | Binary search                     |
| **O(n)**      | 1.000.000                     | Melakukan iterasi array            |
| **O(n log n)**| ~20.000.000                   | Merge Sort, Quick Sort (average)   |
| **O(n²)**     | 1.000.000.000.000             | Bubble Sort, Selection Sort        |
| **O(2ⁿ)**     | Ekspansi eksponensial (~∞)     | Problem kombinatorik seperti brute-force |

- **O(1)** sangat efisien, karena tidak peduli seberapa besar ukuran input, waktu eksekusinya tetap konstan.
- **O(log n)**, meskipun sedikit lebih lambat dari O(1), skalanya tumbuh sangat lambat meskipun ukuran input menjadi sangat besar. Ini mengapa algoritma seperti **binary search** sangat efisien.
- **O(n)** dan **O(n log n)** tumbuh secara signifikan dengan ukuran input yang besar, tetapi masih jauh lebih baik dibandingkan dengan algoritma **O(n²)** atau **O(2ⁿ)**, yang menjadi tidak dapat digunakan untuk input besar.

### 6. **Multi-Variabel Big O**
Dalam beberapa kasus, algoritma memiliki lebih dari satu variabel input yang memengaruhi kompleksitas. Misalnya, dalam algoritma pencarian pada matriks 2D, Anda mungkin memiliki ukuran input yang merepresentasikan jumlah baris dan kolom, `m` dan `n`.

Dalam kasus ini, Big O notation dapat melibatkan dua variabel, seperti **O(m * n)**, yang berarti waktu eksekusi bergantung pada hasil perkalian antara dua variabel.

### 7. **Amortized Analysis**
Untuk beberapa algoritma, kompleksitas waktu rata-rata untuk setiap operasi mungkin rendah, tetapi ada kasus-kasus tertentu di mana waktu eksekusi untuk satu operasi bisa jauh lebih tinggi. **Amortized analysis** digunakan untuk memperhitungkan rata-rata kinerja dari keseluruhan rangkaian operasi.

Contoh umum dari ini adalah dalam **dynamic array** (seperti Python `list` atau Java `ArrayList`), di mana menambahkan elemen baru biasanya dilakukan dalam **O(1)**, tetapi kadang-kadang perlu dilakukan **resizing**, yang membuat operasi penambahan menjadi **O(n)**. Namun, secara keseluruhan, operasi penambahan dianggap **O(1)** dalam skala besar karena resizing terjadi relatif jarang.

### Kesimpulan
Big O notation adalah alat yang sangat kuat untuk mengevaluasi dan membandingkan efisiensi algoritma, terutama ketika ukuran input bertambah besar. Dengan memahami berbagai jenis kompleksitas, seperti O(1), O(n), O(log n), dan O(n²), serta trade-offs antara waktu dan ruang, Anda bisa memilih algoritma yang paling sesuai untuk skenario tertentu.

Mengetahui bagaimana cara berpikir tentang pertumbuhan skala, worst-case versus average-case, dan amortized analysis memberi Anda pemahaman yang lebih dalam tentang performa algoritma dan bagaimana membuat keputusan yang lebih baik dalam pemrograman.