# Ad Auction Simulator

![CI](https://github.com/klepkin-pv/auction-simulator/actions/workflows/ci.yml/badge.svg)

Пет-проект: симулятор рекламного аукциона с ML-оптимизацией ставок.

Создан как демонстрация навыков для позиции DS-стажёра в команде рекламной монетизации.

## Что внутри

| Модуль | Что делает |
|---|---|
| `data/generate.py` | Генерация синтетических данных: рекламодатели, запросы, показы |
| `auction/mechanisms.py` | GSP и VCG аукционы с расчётом цены за клик |
| `models/ctr_model.py` | CTR-предсказание: LogReg (baseline) + нейросеть (PyTorch) |
| `models/bid_optimizer.py` | Оптимизация ставок: аналитическая + градиентная (autograd) |
| `experiments/ab_test.py` | A/B тесты: t-test, bootstrap CI, проверка гипотез |
| `notebooks/analysis.ipynb` | Полный пайплайн с визуализацией и SQL-аналитикой |

## Стек

- Python, NumPy, Pandas, Scikit-learn, PyTorch, SciPy
- SQLite + SQLAlchemy (SQL-аналитика)
- Matplotlib, Seaborn

## Тесты

Автотесты покрывают аукционные механизмы, оптимизатор ставок, A/B-тесты и CTR-модели:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

CI запускает `ruff` и `pytest` в GitHub Actions для каждого `push`/`pull_request`.

## Быстрый старт

```bash
pip install -r requirements.txt

# Генерация данных
python data/generate.py

# Запуск ноутбука
jupyter notebook notebooks/analysis.ipynb
```

### Пример аукциона

Три рекламодателя делают ставки за одно рекламное место:

| Рекламодатель | Ставка | Quality score |
|---|---|---|
| A | 100 ₽ | 0.8 |
| B | 80 ₽ | 1.0 |
| C | 60 ₽ | 0.9 |

**Ранжирование по effective bid** = ставка × quality score:
- B: 80 × 1.0 = 80
- A: 100 × 0.8 = 80
- C: 60 × 0.9 = 54

При равенстве effective bid победитель определяется по ставке. В **GSP** победитель (B) платит достаточную ставку, чтобы удержать позицию:

`price = next_effective_bid / quality_score_winner + epsilon`

Если A занял бы второе место, B заплатил бы примерно 80 ₽.

## Ключевые концепции

**GSP (Generalized Second Price)** — каждый победитель платит ставку следующего конкурента,
скорректированную на quality score. Именно этот механизм используется в большинстве
рекламных платформ.

**CTR-модель** — предсказываем вероятность клика по фичам: позиция, категория, качество
пользователя. Сравниваем логрег и нейросеть по AUC и log-loss.

**Bid Optimization** — находим оптимальную ставку, максимизирующую клики при ограниченном
бюджете. Градиентный подход использует дифференцируемую аппроксимацию аукциона.

**A/B тесты** — проверяем статистическую значимость изменений через t-test и bootstrap.
