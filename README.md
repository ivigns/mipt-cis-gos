# mipt-cis-gos

## Задача №3.1

_Предложите и реализуйте алгоритм, который позволит находить на изображении участки с градиентной заливкой (участки прямоугольной формы, стороны прямоугольника параллельны границам изображения)._

---

## Решение

Под _областью с градиентной заливкой_ будет подразумеваться область изображения, где цвет меняется плавно. То есть в соседних пикселях цвет меняется незначительно. Частный случай градиентной заливки - _линейный градиент_, где цвет меняется вдоль прямой линии от одного к другому.


За основу взят алгоритм поиска наибольшей нулевой подматрицы https://e-maxx.ru/algo/maximum_zero_submatrix и адаптирован под задачу:
* В матрице, полученной применением градиентого фильтра к изображению, ищется наибольшая подматрица с значениями градиента, не превышающими порог.
* Дополнительное условие на матрицу - среднее значение в ней не должно быть меньше порогового. Так мы исключаем из ответа обычную сплошную заливку.

Алгоритм принимает на вход путь до цветного изображения в почти любом формате. Результатом алгоритма является набор двух пар - координаты левого верхнего и правого нижнего конца прямоугольника, где находится градиент на картинке. Для наглядности в папку `rect` сохраняется исходное изображение с результирующим прямоугольником, отмеченным красным. В папку `gradient` сохраняется ч/б изображение - результат применения градиентного фильтра (использовалось для дебага, оставлено для любопытства).

Сложность алгоритма по времени (размер изображения _n x m_):
* Применение градиентного фильтра - _O(nm)_
* Построение матрицы интегральных сумм - _O(nm)_
* Алгоритм поиска наибольшей подматрицы "с маленькими значениями" градиента - _O(nm)_
  * Предпосчет - _O(nm)_
  * Поиск границ подматрицы при фиксированных i, j - _O(1)_
  * Поиск суммы подматрицы при фиксированных i, j - _O(1)_

Итого: _O(nm)_

Использовано доп памяти _O(nm)_.

## Запуск

Для работы необходим Python 3.7.9, opencv2, numpy.

```bash
python3 detector.py <путь до изображения>
```

При запуске на некоторых `png` изображениях может возникать следующая ошибка:

```
libpng warning: iCCP: known incorrect sRGB profile
```

Проблема в этом случае в самом изображении, его нужно предобработать следующей командой:

```bash
mogrify *.png
```

## Конфигурация

Некоторые параметры можно изменять через конфигурационный файл `config.json`:
* **gradient_dir** - путь до папки, куда будет сохраняться изображение после применения градиентного фильтра (можно не указывать),
* **rect_dir** - путь до папки, куда будет сохраняться исходное изображение с отмеченным результирующим прямоугольником,
* **grad_max** - максимальное допустимое значение градиента изображения, чтобы заливка прямоугольника считалась градиентной,
* **min_avg** - минимальное среднее значение градиента в результирующем прямоугольнике (для отбрасывания прямоугольников со сплошной заливкой, можно указать -1, чтобы этого не делать),
* **rect_color** - цвет результирующего прямоугольника в формате `[B, G, R]`,
* **rect_thickness** - толщина границы результирующего прямоугольника в пикселях.
