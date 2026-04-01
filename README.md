# Ad Auction Simulator

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

## Быстрый старт

```bash
pip install -r requirements.txt

# Генерация данных
python data/generate.py

# Запуск ноутбука
jupyter notebook notebooks/analysis.ipynb
```

## Ключевые концепции

**GSP (Generalized Second Price)** — каждый победитель платит ставку следующего конкурента,
скорректированную на quality score. Именно этот механизм используется в большинстве
рекламных платформ.

**CTR-модель** — предсказываем вероятность клика по фичам: позиция, категория, качество
пользователя. Сравниваем логрег и нейросеть по AUC и log-loss.

**Bid Optimization** — находим оптимальную ставку, максимизирующую клики при ограниченном
бюджете. Градиентный подход использует дифференцируемую аппроксимацию аукциона.

**A/B тесты** — проверяем статистическую значимость изменений через t-test и bootstrap.
