
Paradigma pemrograman bukanlah bahasa atau alat. Anda tidak dapat "membangun" apa pun dengan paradigma. Paradigma lebih seperti sekumpulan cita-cita dan pedoman yang telah disetujui, diikuti, dan dikembangkan oleh banyak orang.

## Paradigma Fungsional
Paradigma ini fokus pada **fungsi** sebagai unit utama dalam menyusun program. Fungsi dianggap sebagai pemetaan dari input ke output dan sebaiknya **tidak memiliki efek samping** (fungsi murni). Program dibangun dengan **komposisi fungsi** di mana output dari satu fungsi bisa menjadi input bagi fungsi lain. Karakteristik lain termasuk **immutability** (data tidak bisa diubah setelah dibuat) dan **rekursi** (pengulangan dilakukan melalui pemanggilan diri sendiri, bukan dengan loop).

#### Ciri-ciri:

- Menghindari perubahan state atau data mutable.
- Lebih fokus pada **apa** yang ingin dicapai (deklaratif).
- Sering menggunakan rekursi alih-alih loop.
- Fungsi murni: hasil fungsi hanya ditentukan oleh inputnya dan tidak bergantung pada state eksternal.

*Contoh: Menghitung kuadrat dari sebuah list angka di Python (paradigma fungsional)*:

```python
numbers = [1, 2, 3, 4]  
# Menggunakan fungsi map (fungsi murni) 
squares = list(map(lambda x: x ** 2, numbers))  
print(squares)
```

## Paradigma Prosedural
Paradigma ini fokus pada **urutan instruksi** yang dijalankan untuk menyelesaikan suatu tugas. Dalam paradigma ini, program dipandang sebagai serangkaian **prosedur** (atau subrutin) yang menjalankan langkah-langkah tertentu untuk mengolah data. Prosedur dapat mengubah state program, misalnya dengan memodifikasi variabel global.

#### Ciri-ciri:

- Memiliki **urutan langkah** atau instruksi yang dijalankan secara eksplisit.
- Memodifikasi **state program** melalui variabel yang dapat berubah.
- Menggunakan loop dan pernyataan kondisional untuk kontrol alur.
- Lebih fokus pada **bagaimana** melakukan sesuatu (imperatif).

*Contoh: Menghitung kuadrat dari sebuah list angka di Python (paradigma prosedural):*
```python
numbers = [1, 2, 3, 4]
squares = []

# Menggunakan loop (gaya prosedural)
for number in numbers:
    squares.append(number ** 2)

print(squares)

```

### Perbedaan Utama Fungsional dan Prosedural
- **State dan Mutability**: Paradigma fungsional cenderung menghindari perubahan state dan bekerja dengan data immutable, sementara paradigma prosedural sering memodifikasi state menggunakan variabel mutable.
- **Pendekatan Pemrograman**: Pemrograman fungsional lebih deklaratif, menjelaskan **apa** yang dilakukan fungsi. Sebaliknya, pemrograman prosedural lebih imperatif, menjelaskan **bagaimana** tugas dilakukan langkah demi langkah.
- **Rekursi vs Loop**: Pemrograman fungsional cenderung menggunakan rekursi sebagai cara untuk melakukan pengulangan, sementara prosedural lebih sering menggunakan loop.

# Bahasa Pemrograman dengan Paradigma

Beberapa bahasa pemrograman dirancang dengan paradigma tertentu sebagai fokus utama, meskipun banyak bahasa modern mendukung **multi-paradigma**, memungkinkan penggunaan berbagai gaya pemrograman. Berikut beberapa contoh bahasa pemrograman yang erat terkait dengan paradigma tertentu:

### 1. **Bahasa Fungsional**
Bahasa pemrograman fungsional dirancang untuk mengikuti paradigma fungsional secara ketat.

- **Haskell**: Haskell adalah bahasa pemrograman fungsional murni, di mana segala sesuatu adalah fungsi, dan fungsi murni tidak memiliki efek samping. Haskell memiliki tipe statis yang kuat dan mendukung konsep **lazy evaluation** (evaluasi malas), artinya ekspresi tidak akan dievaluasi sampai hasilnya dibutuhkan.
  
  Contoh kode (fungsi penjumlahan):
  ```haskell
  sumList :: [Int] -> Int
  sumList [] = 0
  sumList (x:xs) = x + sumList xs
  ```

- **Lisp**: Lisp adalah salah satu bahasa fungsional tertua dan sangat fleksibel. Meskipun tidak selalu murni fungsional, ia mendukung gaya pemrograman fungsional dan pemrosesan daftar (list processing).

  Contoh kode (fungsi faktorial):
  ```lisp
  (defun factorial (n)
    (if (<= n 1)
        1
        (* n (factorial (- n 1)))))
  ```

- **Erlang**: Dirancang untuk aplikasi yang sangat reliabel dan paralel, Erlang mendorong penggunaan fungsi murni dan berkomunikasi melalui pesan tanpa berbagi memori, yang sangat sejalan dengan pemrograman fungsional.

  Contoh kode (fungsi rekursif menghitung panjang daftar):
  ```erlang
  length([]) -> 0;
  length([_ | T]) -> 1 + length(T).
  ```

### 2. **Bahasa Prosedural**
Bahasa pemrograman prosedural fokus pada urutan langkah-langkah dan aliran kontrol program.

- **C**: C adalah bahasa prosedural yang sangat populer dan sering digunakan untuk pemrograman sistem. Program ditulis sebagai serangkaian prosedur atau fungsi, dan variabel dapat dimodifikasi secara bebas.
  
  Contoh kode (loop menghitung faktorial):
  ```c
  int factorial(int n) {
      int result = 1;
      for (int i = 1; i <= n; i++) {
          result *= i;
      }
      return result;
  }
  ```

- **Pascal**: Pascal adalah bahasa yang dikembangkan untuk mendukung pemrograman prosedural dan pembelajaran pemrograman. Struktur program didasarkan pada prosedur dan fungsi yang disusun secara hierarkis.
  
  Contoh kode (program menghitung faktorial):
  ```pascal
  function factorial(n: integer): integer;
  var
      i, result: integer;
  begin
      result := 1;
      for i := 1 to n do
          result := result * i;
      factorial := result;
  end;
  ```

### 3. **Bahasa Berorientasi Objek**
Bahasa ini didesain untuk mendukung paradigma **berorientasi objek** (OOP), meskipun sebagian besar juga bisa digunakan dalam gaya prosedural.

- **Java**: Bahasa pemrograman berorientasi objek yang menekankan penggunaan kelas dan objek untuk mendefinisikan struktur program. Semua kode di Java harus berada dalam sebuah kelas.
  
  Contoh kode (kelas `Person` dengan metode):
  ```java
  class Person {
      private String name;
      
      public Person(String name) {
          this.name = name;
      }

      public void greet() {
          System.out.println("Hello, my name is " + name);
      }
  }
  ```

- **C++**: C++ adalah bahasa yang mendukung baik paradigma berorientasi objek maupun prosedural. Kelebihan C++ adalah fleksibilitasnya dalam memungkinkan pemrogram menulis program dengan berbagai gaya.

  Contoh kode (kelas dengan metode):
  ```cpp
  class Person {
      private:
          std::string name;
      public:
          Person(std::string n) : name(n) {}
          void greet() {
              std::cout << "Hello, my name is " << name << std::endl;
          }
  };
  ```

### 4. **Bahasa Multi-Paradigma**
Beberapa bahasa mendukung kombinasi paradigma (fungsional, prosedural, objek, dll.) dan memberikan fleksibilitas kepada programmer.

- **Python**: Python adalah bahasa yang mendukung berbagai paradigma, termasuk prosedural, fungsional, dan berorientasi objek. Pengembang dapat memilih pendekatan mana yang lebih sesuai untuk masalah yang mereka selesaikan.

  Contoh kode prosedural:
  ```python
  def factorial(n):
      result = 1
      for i in range(1, n + 1):
          result *= i
      return result
  ```

  Contoh kode berorientasi objek:
  ```python
  class Person:
      def __init__(self, name):
          self.name = name

      def greet(self):
          print(f"Hello, my name is {self.name}")
  ```

  Contoh kode fungsional:
  ```python
  numbers = [1, 2, 3, 4]
  squares = list(map(lambda x: x ** 2, numbers))
  ```

- **Scala**: Scala mendukung paradigma fungsional dan berorientasi objek, sering digunakan dalam konteks aplikasi skala besar seperti yang ditangani oleh platform big data seperti Apache Spark.

  Contoh kode (fungsional dan objek):
  ```scala
  val numbers = List(1, 2, 3, 4)
  val squares = numbers.map(x => x * x)
  ```

### Kesimpulan:
- **Bahasa Fungsional**: Haskell, Lisp, Erlang.
- **Bahasa Prosedural**: C, Pascal.
- **Bahasa Berorientasi Objek**: Java, C++.
- **Bahasa Multi-Paradigma**: Python, Scala.

Banyak bahasa modern bersifat multi-paradigma, sehingga programmer dapat mengadopsi berbagai pendekatan sesuai dengan kebutuhan.

[reference](https://www.freecodecamp.org/news/an-introduction-to-programming-paradigms/)
