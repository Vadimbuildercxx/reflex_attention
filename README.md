
# Reflex attention

Имплементация reflex attention в модели NanoGPT.

## Install

## Reflex / Route attention

По скольку трансформеры склонны при увеличении контекста схлопывать репрезентации.
Попробуем имплементировать Reflex attention посчитав cross-attention (CA) по предыдущим слоям и (SA) по конкретному слою.

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

Однако стоит заметить что модель не была обучена до конца. Поэтому поступим следующим образом:

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

Также модель с роутером обученная на умножении чисел похожа на модель обученной на `openwebtext`, но по скольку была возможность обучить модель на большем количестве шагов мы можем также посмотреть на веса в $K_{router}$ и $V_{router}$.

![alt text](images/layers_connection_compare_mul.png)

В данном случае видно, что в то время как $K_{router}$ обращает меньшее внимание на прошлые токены, $V_{router}$ склонен обращать внимание и на предыдущие токены в том числе. Также можно заметить что подача дополнительных токенов имеет большее влияние на слои находящиеся в середине.

Попробуем посмотреть на энтропию самих скрытых состояний токенов. Для этого прогоним часть сэмплов через модель и получим из каждого скрытые состояния токенов. И далее для каждого посчитаем энтропию Реньи по следующей формуле:

$\mathcal{H_i}_\alpha(X) = \frac{1}{1-\alpha} \log \left( \sum_{j=1}^n p_j^\alpha \right), i \in (1, ..., n)$, 

где вероятность считается следующим образом $p_j= \frac{\lambda_j(H_i \cdot H_i^T)}{tr(H_i \cdot H_i^T)}$ и $H_i \in \mathbb{R}^{T \times d}$ - скрытое состояние на $i$-том слое, $\lambda_j$-собственные значение матрицы $H_i \cdot H_i^T$.

Посчитаем для разных длин последовательностей энтропию слоев и сравним ее с моделью с обычным Self-Attention.

![alt text](images/layers_entropy.png)

Как можем видеть на картинке выше, в средних слоях энтропия особенно выше у модели с роутами, что говорит как о том что роуты влияют на репрезентации средних слоев, так и о том что они влияют на скорость сходимости модели. Также можем заметить что именно $K_{router}$ и $V_{router}$ в средних слоях склонны обращать больше внимания на поданные репрезентации с прошлых слоев. Ниже преведено сравнение средних значений энтропии у двух моделей. 

![alt text](images/layers_entropy_simple.png)

Такое же поведение наблюдается и у моделей с более длинной последовательностью.

![alt text](images/layers_entropy_long.png)

## Выводы


## Установка

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3





If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-shakespeare-char
```

This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after 3 minutes of training on a GPU. Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section later).

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~3 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
Generates samples like this:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `python sample.py`.

Finally, to train on a single GPU simply run the `python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. For an example of how to finetune a GPT on new text go to `data/shakespeare` and run `prepare.py` to download the tiny shakespeare dataset and render it into a `train.bin` and `val.bin`, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText this will run in seconds. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```sh
python train.py config/finetune_shakespeare.py
```

This will load the config parameter overrides in `config/finetune_shakespeare.py` (I didn't tune them much though). Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-shakespeare` by default, per the config file. You can then run the code in `sample.py --out_dir=out-shakespeare`:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

Whoa there, GPT, entering some dark place over there. I didn't really tune the hyperparameters in the config too much, feel free to try!

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.

## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!
