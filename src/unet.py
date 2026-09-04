"""UNet для восстановления одиночного кадра.

Архитектура намеренно стандартная и заморожена: она общая для всех
16 ячеек матрицы. Изменение здесь обесценивает уже посчитанные ячейки,
потому что различия между ними перестанут объясняться только данными.

Три отличия от Стандартного UNet, все общеприняты в restoration:

  * GroupNorm вместо BatchNorm. BatchNorm хранит running mean/var,
    накопленные на обучающем распределении. Модель, обученная на D0
    и прогнанная по D3, просела бы отчасти из-за рассогласования этих
    статистик, а не из-за непереносимости признаков — то есть
    внедиагональные ячейки мерили бы не то, ради чего считаются.
  * Билинейный апсемплинг + свёртка 3x3 вместо ConvTranspose:
    транспонированная свёртка даёт регулярные шахматные артефакты.
  * Выход линейный, без sigmoid и clamp: sigmoid насыщается и сжимает
    динамический диапазон у самых краёв, где и живёт часть деталей,
    а clamp обнуляет градиент на вылетевших пикселях. Обрезка в [0, 1]
    делается ровно один раз, в metrics.py, и там же объяснено почему:
    пиксель, вылезший за диапазон, при сохранении в uint8 исчезает.

Вход  (B, 1, H, W) float32, значения ~[0, 1]
Выход (B, 1, H, W) float32, диапазон не ограничен

H и W произвольные: forward сам дополняет до кратности 2**depth
и обрезает результат обратно. Это нужно не на обучении (кроп 128
кратен 8), а на OTIS, где кадры бывают вроде 135x135.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels, groups=8):
    """GroupNorm; число групп ужимается, если не делит число каналов."""
    while channels % groups and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def _conv(in_ch, out_ch):
    """Свёртка 3x3 -> норма -> ReLU. bias не нужен: его съедает норма."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        _norm(out_ch),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    """Две свёртки 3x3 — базовый блок каждого уровня."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(_conv(in_ch, out_ch), _conv(out_ch, out_ch))

    def forward(self, x):
        return self.body(x)


class Up(nn.Module):
    """Апсемплинг до размера skip-связи, сжатие каналов, конкатенация.

    Интерполяция идёт к точному размеру skip, а не scale_factor=2:
    при нечётной стороне после паддинга размеры разошлись бы на пиксель
    и torch.cat упал бы.
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = _conv(in_ch, out_ch)
        self.fuse = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.reduce(x), skip], dim=1))


class UNet(nn.Module):
    """depth — число пулингов; base — ширина первого уровня."""

    def __init__(self, in_ch=1, out_ch=1, base=48, depth=3):
        super().__init__()
        self.depth = int(depth)
        w = [base * 2 ** i for i in range(depth + 1)]

        self.inc = DoubleConv(in_ch, w[0])
        self.pool = nn.MaxPool2d(2)
        self.downs = nn.ModuleList([DoubleConv(w[i], w[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList(
            [Up(w[i + 1], w[i], w[i]) for i in reversed(range(depth))]
        )
        self.head = nn.Conv2d(w[0], out_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        #Голову нужно отдельно, така как kaiming ждёт, что после слоя идет Relu
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, 0.5)

    def forward(self, x):
        h, w = x.shape[-2:]
        mult = 2 ** self.depth
        pad_h, pad_w = (-h) % mult, (-w) % mult
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        feats = [self.inc(x)]
        for down in self.downs:
            feats.append(down(self.pool(feats[-1])))

        y = feats[-1]
        for up, skip in zip(self.ups, reversed(feats[:-1])):
            y = up(y, skip)

        return self.head(y)[..., :h, :w]


def receptive_field(depth=3):
    """Рецептивное поле выхода по входу, в пикселях.

    r — текущее поле, j — шаг между соседними позициями фичемапы
    в пикселях входа. Свёртка ядром k добавляет (k-1)*j, пулинг 2x2
    со stride 2 добавляет j и удваивает j, апсемплинг делит j пополам.
    Считается максимум по путям, то есть самый глубокий путь;
    skip-связи дают более короткие пути с меньшим полем.
    """
    r, j = 1, 1
    r += 2 * 2 * j                     # inc: две свёртки 3x3
    for _ in range(depth):
        r += j
        j *= 2                         # MaxPool 2x2 stride 2
        r += 2 * 2 * j                 # DoubleConv уровня
    for _ in range(depth):
        j //= 2                        # билинейный апсемплинг x2
        r += j                         # интерполяция смешивает 2 соседа
        r += 2 * j                     # свёртка сжатия каналов
        r += 2 * 2 * j                 # DoubleConv после конкатенации
    return r                           # head 1x1 ничего не добавляет


if __name__ == "__main__":
    net = UNet()
    n = sum(p.numel() for p in net.parameters())
    print(f"параметров: {n / 1e6:.2f} M, RF: {receptive_field(net.depth)} px")
    for size in (128, 135, 256):
        y = net(torch.zeros(2, 1, size, size))
        print(f"вход {size}x{size} -> выход {tuple(y.shape)}")
