# Reflex attention

Известно, что, так как, декодеры обучены предсказывать только следующие токены (в отличие от энкодеров), модель страдает от отстутствия прямого доступа к репрезентациям предыдущих слоев, что сказывается на ухудшении работы модели как при большом контексте, так и при увеличениии количества слоев.
Попробуем решать данную проблему, рассмотря reflex attention и route attention добавленные в модель NanoGPT.

## Установка

```
pip install torch numpy transformers datasets tiktoken wandb tqdm scikit-learn
```

## Reflex / Route attention

По скольку трансформеры склонны при увеличении контекста схлопывать репрезентации,
попробуем имплементировать Reflex attention посчитав cross-attention (CA) по предыдущим слоям и (SA) по конкретному слою.

Зададим внимание следующим образом:

$Attn_i = Cat[SA(h_i), \; CA(h_{i-1}, h_i) , \; CA(h_{i-2}, h_i)]$

Также попробуем задать его модифицированную версию. Добавим $K_{router}$ и $V_{router}$ И будем получать $K$ и $V$ помощью линейной комбинации с предыдущих выходов модели.

$K = \sum_{i=0}^{H \cdot L} \alpha_i \cdot K_i, \alpha_i \in K_{router}$,

$V = \sum_{i=0}^{H \cdot L} \beta_i \cdot V_i, \beta_i \in V_{router}$,

где $H$ - количество голов внимания, $L$ - количество скрытых подаваемых скрытых состояний + вход слоя. И посчитаем внимание как обычно.

$Attn = SA[Q, K, V]$.

Как можно увидеть на рисунке ниже, есть улучшения по сравнению с классической моделью loss стал падать быстрее у модели с reflex attention и еще немного быстрее у модели с роутом. 

![alt text](images/reflex_n_route_attn_compare.png)

 
По скольку количество слоев не такое большое, и коллапс репрезентаций происходит при увеличении слоев [[Transformers need glasses](https://arxiv.org/pdf/2406.04267)] попробуем увеличить количество слоев до 18. И проверим работу модели.
Также уменьшим размер контекстного окна чтобы модель поместилась в память. Можем увидеть результат на следующем графике: 

![alt text](images/l18_reflex_n_route_attn_compare.png)

Как можно видеть модель с роутингом показала себя неплохо и здесь, однако обычный reflex attention показал себя хуже.

Посмотрим в роутере важность каждого из Q и K значений. Вытащим из них веса и сопоставим их выходам с других слоев.
На изображении ниже, можем увидеть все веса роутер сгрупированные по слоям и по типу роутера. По оси ордиат обозначены поданные на вход слои для данного слоя. По оси абсцисс наименования голов. 

Как можем видеть на изображении роутеры более склонны обращать внимание на предыдущие токены на средних слоях (в данном случае это слои 3-4). Также можно заметить что $V_{router}$ более склонен к перебалансировке весов из предыдущих токенов.

![alt text](images/layers_connection_compare.png)

Однако стоит заметить что модель не была обучена до конца, по скольку это требует достаточно много времени. Поэтому поступим следующим образом:

## Дополнительно
По скольку в трансформерах имеет место коллапс репрезентаций, что ведет к ухудшению качества модели при увеличении последовательности попробуем для решения этой задачи применить Reflex attention. Попробуем обучить NanoGPT с reflex-attention и классическим attention на умножении четырехзначных чисел [SEQ-VCR:PREVENTING COLLAPSE IN INTERMEDIATE
TRANSFORMER REPRESENTATIONS FOR ENHANCED
REASONING](https://arxiv.org/pdf/2411.02344) и сравним их качество.

Токенизатор использовался стандартный как для GPT 2.

Пример последовательностей:
```
'6322, 468966, 468966, 0, 234483] => 2396572582\n99288*37728 = [794304, 198576, 695016, 695016, 297864] => 3745937664\n71142*28813 = [213426, 71142, 569136, 569136, 142284] => 2049814446\n56112*18452 = [112224, 280560, 224448, 448896, 56112] => 1035378624\n51994*48954 = [207976'

'22, 468966, 468966, 0, 234483] => 2396572582\n99288*37728 = [794304, 198576, 695016, 695016, 297864] => 3745937664\n71142*28813 = [213426, 71142, 569136, 569136, 142284] => 2049814446\n56112*18452 = [112224, 280560, 224448, 448896, 56112] => 1035378624\n51994*48954 = [207976,'
```

Поведение обучения модели можем пронаблюдать на графике ниже. Видно что модель с обычным Reflex Attention перестает обучаться (примерно на пятом шаге) в то время как ее модификация с $K_{router}$ и $V_{router}$ обгоняет модель с классическим SA. 

![alt text](images/l6_reflex_n_route_attn_compare_digit_multiplication.png)

Также модель с роутером обученная на умножении чисел имеет схожесть в весах роута модели обученной на `openwebtext`, но по скольку была возможность обучить модель на большем количестве шагов мы можем также посмотреть на веса в $K_{router}$ и $V_{router}$ более точно.

![alt text](images/layers_connection_compare_mul.png)

В данном случае видно, что в то время как $K_{router}$ обращает меньшее внимание на прошлые токены, $V_{router}$ склонен обращать внимание и на предыдущие токены в том числе. Также можно заметить что подача дополнительных токенов имеет большее влияние на слои находящиеся в середине.

Попробуем посмотреть на энтропию самих скрытых состояний токенов. Для этого прогоним часть сэмплов через модель и получим из каждого скрытые состояния токенов. И далее для каждого посчитаем энтропию Реньи по следующей формуле:

$$\Eta_{i,\alpha}(X) = \frac{1}{1-\alpha} \log\left(\sum_{j=1}^n p_j^\alpha \right), \; i \in \{1, \dots, n\}$$

, где вероятность считается следующим образом $p_j= \frac{\lambda_j(H_i \cdot H_i^T)}{tr(H_i \cdot H_i^T)}$ и $H_i \in \mathbb{R}^{T \times d}$ - скрытое состояние на $i$-том слое, $\lambda_j$-собственные значение матрицы $H_i \cdot H_i^T$.


Посчитаем для разных длин последовательностей энтропию слоев и сравним ее с моделью с обычным Self-Attention.

![alt text](images/layers_entropy.png)

Как можем видеть на картинке выше, в средних слоях энтропия особенно выше у модели с роутами, что говорит как о том что роуты влияют на репрезентации средних слоев, так и о том что они влияют на скорость сходимости модели. Также можем заметить что именно $K_{router}$ и $V_{router}$ в средних слоях склонны обращать больше внимания на поданные репрезентации с прошлых слоев. Ниже преведено сравнение средних значений энтропии у двух моделей. 

![alt text](images/layers_entropy_simple.png)

Такое же поведение наблюдается и у моделей с более длинной последовательностью.

![alt text](images/layers_entropy_long.png)

Можем также посмотреть на точность вычислений и примеры поведения двух моделей.

|| $R^2$ | $MAE$  |
|---|-------|------|
|$Attn_{classic}$|   0.8767    |    1855158.71  |
|$Attn_{route}$|   0.9877    |     970777.22  |

## Выводы
Судя по экспериментам, при подаче дополнительных скрытых состояний, у модели действительно наблюдаются улучшения. Модель начинает быстрее сходится. Однако Reflex attetion в отличие от его модификации работает значительно хуже при дальнейшей тренировке. Возможно это связано с его имплементацией и там явно имеют место некоторые доработки. 

Из интересного можно отметить что дополнительные скрытые состояния на входе повышают энтропию на средних слоях модели. Что, возможно, лучше сказывается на качестве модели. Мы могли это увидеть как на графиках обучения так и на самих графиках энтропии. И также, роутеры в средних слоях, склонна обращать внимание к предыдущим выходным токенам больше, чем роутеры в первых или последних слоях.

## Послесловие / Дискуссия
По скольку Route attention дал неплохой прирост в характеристиках модели, можно было бы протестировать его на бОльшем количестве слоев и посмотреть будет ли наблюдаться данный эффект там, по скольку данном случае ($Layers=6, Heads=6$) модель достаточно мала, чтобы на ней проявлялись все недостатки больших трансформеров. Также можно было бы обучить более большую модель для решения математических выражений.