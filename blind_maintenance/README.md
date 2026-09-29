# Blind Maintenance Benchmark — независимая реализация (пакет A, v1.0)

Независимая реализация спецификации `protocol/Blind_Maintenance_Benchmark_RU_v1.md`:
синтетическая среда обслуживания устройства, контроллер C, его варианты и сравнительные
стратегии. Исходный код авторов, их публикации и результаты не использовались.

## Установка

```bash
python3 -m pip install -r requirements.txt     # numpy, pytest; Python 3.11
```

## Команды

| шаг | команда | что делает |
|---|---|---|
| тесты §8 | `python3 -m pytest -q tests --junitxml results/tests/junit.xml` | проверки на заданных вручную состояниях и лентах |
| smoke-test | `python3 scripts/smoke_test.py` | 2+2 development-мира; это НЕ эксперимент |
| development | `python3 scripts/run_development.py` | 64+64 мира, сетки настройки, `frozen/settings.json` |
| анализ dev | `python3 scripts/analyze.py --phase development` | описательная сводка development |
| заморозка | `python3 scripts/freeze.py` | `frozen/manifest.json`: SHA-256 кода и настроек, версии, RNG, метрики |
| seeds | `python3 scripts/draw_confirmatory_seeds.py [--master-seed N]` | master seed строго после заморозки; 512 dynamic + 256 static миров |
| итоговый запуск | `python3 scripts/run_confirmatory.py` | сверяет хеши, запускает прогон один раз, пишет пошаговые логи |
| анализ | `python3 scripts/analyze.py --phase confirmatory` | `summary.json`, `summary_tables.md` |

Полный запуск с нуля занимает около 2 минут на 4 ядрах:

```bash
python3 -m pytest -q tests --junitxml results/tests/junit.xml
python3 scripts/run_development.py && python3 scripts/analyze.py --phase development
python3 scripts/freeze.py
python3 scripts/draw_confirmatory_seeds.py
python3 scripts/run_confirmatory.py && python3 scripts/analyze.py --phase confirmatory
```

`freeze.py`, `draw_confirmatory_seeds.py` и `run_confirmatory.py` отказываются
перезаписывать уже созданные файлы. Повторный итоговый прогон возможен только с
`--rerun-reason`: причина записывается в `CHANGELOG.md`, старые результаты сохраняются.

Чтобы воспроизвести опубликованный итоговый прогон, оставьте `frozen/` как есть и
запустите `run_confirmatory.py` в копии, где `results/confirmatory/` удалён. Числа
должны совпасть побитово: все случайные величины выводятся из seed в `frozen/`.

## Структура

```
bm/params.py        константы спецификации
bm/tape.py          экзогенная лента (RNG-потоки, доступность, износ, шумы, π, знаки, q_t)
bm/world.py         скрытое состояние, физика шага, API наблюдений, привилегированные исключения
bm/policies.py      work-only, fixed-p, output-only, tuned-output, output-trend, C, C-scrambled, C-level, Oracle-D
bm/runner.py        эпизод: мир ↔ политика, метрики, пошаговый лог анализатора
bm/stats.py         парный percentile bootstrap, Wilson
bm/seeds.py         вывод confirmatory-миров из master seed
scripts/            запуск фаз, заморозка, анализ
tests/              проверки §8
protocol/           спецификация (копия) и решения реализации
frozen/             settings.json, manifest.json, confirmatory_seeds.json
results/            development/, confirmatory/ (csv, summary.json, summary_tables.md, step_logs/), tests/
REPORT_RU.md        отчёт
CHANGELOG.md        журнал изменений
```
