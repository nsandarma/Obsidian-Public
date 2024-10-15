# Array
Perbedaan Utama:
1. Passing by value tidak benar-benar berlaku untuk array di C karena Anda tidak bisa mengirim seluruh array. C secara otomatis akan mengonversi array menjadi pointer ketika Anda melewatkannya ke fungsi.
2. Passing by reference secara default berlaku untuk array karena yang dikirimkan adalah pointer ke elemen pertama array, memungkinkan fungsi untuk memodifikasi array asli.
Untuk array, C selalu menggunakan passing by reference secara implisit karena yang dilewatkan adalah pointer ke array, bukan salinan seluruh array.

```c
#include <stdio.h>

void modifyArray(int arr[], int size) {
    arr[0] = 100;  // Mengubah elemen pertama array
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    modifyArray(arr, 5);
    printf("%d\n", arr[0]);  // Output: 100 (nilai array asli berubah)
    return 0;
}
```

Meskipun sepertinya kita "mem-pass" array, sebenarnya kita sedang mengirimkan **pointer ke array**. Karena array dipass-by-reference secara implisit, perubahan pada elemen array di dalam fungsi akan memengaruhi array asli.

# Struct
berbeda dengan di struct pass by value dan pass by reference adalah kedua hal yang berbeda , dimana pass by value akan menyalin sedangkan pass by reference akan melewatkan alamat nya saja alias pointer

Ketika kamu memiliki sebuah `struct` di C yang berisi pointer ke array, kamu bisa memutuskan untuk melewatkan `struct` tersebut ke fungsi baik secara langsung (by value) maupun menggunakan pointer (by reference). 

Berikut perbedaannya:

### 1. **Passing `struct` secara langsung (by value):**
   - **Contoh**: `void myFunction(struct MyStruct s)`.
   - Dalam hal ini, seluruh isi dari `struct` akan disalin, termasuk pointer yang ada di dalam `struct`. Ini berarti fungsi akan menerima salinan dari `struct` tersebut.
   - **Kelemahan**: Proses penyalinan bisa memakan waktu jika `struct`-nya besar. Selain itu, perubahan yang dilakukan pada salinan di dalam fungsi tidak akan mempengaruhi `struct` asli di luar fungsi.

   **Contoh**:
   ```c
   struct MyStruct {
       int *array;
   };

   void myFunction(struct MyStruct s) {
       // Mengubah array di dalam struct hanya mengubah salinannya
       s.array[0] = 42;
   }
   ```

### 2. **Passing `struct` menggunakan pointer (by reference):**
   - **Contoh**: `void myFunction(struct MyStruct *s)`.
   - Dalam hal ini, fungsi akan menerima alamat dari `struct` yang asli. Artinya, perubahan yang dilakukan pada `struct` di dalam fungsi akan mempengaruhi `struct` yang asli.
   - **Kelebihan**: Tidak ada penyalinan besar-besaran karena hanya alamat memori yang dilewatkan, sehingga lebih efisien untuk `struct` besar.
   - **Kekurangan**: Kamu harus menggunakan notasi pointer (`->`) untuk mengakses anggota `struct`.

   **Contoh**:
   ```c
   struct MyStruct {
       int *array;
   };

   void myFunction(struct MyStruct *s) {
       // Mengubah array di dalam struct akan mempengaruhi struct asli
       s->array[0] = 42;
   }
   ```

### Kesimpulan:
- Jika kamu ingin menghemat memori dan performa, serta ingin perubahan yang dilakukan di dalam fungsi mempengaruhi `struct` asli, maka gunakan pointer (`struct MyStruct *s`).
- Jika kamu tidak ingin perubahan pada `struct` asli atau `struct`-nya kecil dan kamu tidak khawatir tentang overhead dari penyalinan, maka passing by value bisa digunakan.

Mana yang kamu pilih tergantung pada kebutuhan fungsionalitas dan efisiensi programmu.
