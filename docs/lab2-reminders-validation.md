# Напоминания и /timezone — проверка 16.11.2026

Пользователь запросил реализацию сохранения напоминаний с подтверждением и личной
IANA-зоны. Доставка в назначенное время остаётся этапом лабораторной № 3.

## Ответственности и хранение

- ReminderStore владеет настройками, подготовленными действиями и записями PostgreSQL.
  Миграция CREATE TABLE/INDEX IF NOT EXISTS сохраняет существующие данные.
- ScheduleAgent выбирает обнаруженный add_reminder через native tool calling,
  валидирует draft и подготавливает действие. До кнопки MCP-записи нет.
- Telegram handler показывает нормализованные параметры и связывает callback с
  владельцем. При подтверждении передаёт действие через подписанный MCP-контекст.
- MCP-сервер проверяет подпись, scope, аргументы и confirmed-действие в БД;
  INSERT и переход executed выполняются атомарно. UNIQUE и row lock защищают повтор.

## Требования и доказательства

| Требование | Проверки |
|---|---|
| N1: личная сохранённая IANA-зона | test_reminders.py, test_reminder_handlers.py; отдельный PostgreSQL с UTC/Asia/Tokyo, повторное соединение |
| N2: native выбор и запрет owner/model metadata | test_reminder_agent.py; MCP без подписи и с чужим owner получает отказ |
| N3: будущее время, offset, DST | test_reminders.py: gap/fold/wrong offset; PostgreSQL future recheck перед INSERT |
| N4: TTL, неизменность, подтверждение | unit/callback tests; PostgreSQL expired/cancelled/changed payload; ожидание row lock не продлевает TTL |
| N5: изоляция и конкурентные повторы | две параллельные prepare и два MCP-confirmed execute дают одно действие и одну запись |
| N6: достоверный результат | callbacks проверяют ID, text, ISO-момент и зону; ошибки не сообщаются как успех |

## Выполненные проверки

- `python -B -m pytest -q`: **382 passed, 5 skipped**, 113,47 s.
  Пять skips — Docker/PostgreSQL-проверки, включаемые отдельно.
- `RUN_INTEGRATION=1 python -B -m pytest -q tests/test_reminder_integration.py
  tests/test_personal_schedule_integration.py`: **3 passed**, 88,27 s. Каждый тест
  использовал отдельный временный контейнер/volume, не рабочую БД. Два прежних
  интеграционных теста инфраструктуры из test_integration.py в этот прогон не включались.
- `python -m ruff check .`, `python -m ruff format --check .`, `git diff --check`: PASS.
- После последней правки инструкции/описания: `python -B -m pytest -q
  tests/test_reminder_agent.py tests/test_reminders.py tests/test_reminder_handlers.py
  tests/test_schedule_mcp.py` — **84 passed**, 74,52 s.
- Независимое чтение storage, host, MCP и handlers: блокирующих замечаний нет.
  Исправлено использование времени до долгого LLM-ответа: prepare в production
  читает актуальные часы, фиксированное now используется только при явной подстановке.

## Проверка реальной модели

Синтетический запрос «Напомни завтра в 18:00 сделать домашку», fixed now
2026-11-16T12:00:00Z, без Telegram и записей в БД:

- Без сохранённой зоны: PASS — модель просит /timezone, prepare не выполняется.
- Первоначальная проверка с Europe/Moscow выявила текстовое «всё верно?» вместо
  native вызова. В инструкции и описании инструмента уточнено: при известных
  параметрах сразу предложить вызов, затем приложение перехватит его и покажет кнопки.
- Повтор с настоящей схемой из server.list_tools: PASS в пустой истории и после
  старого текстового подтверждения. Оба раза PendingReminder с
  remind_at=2026-11-17T18:00:00+03:00 и текстом «Сделать домашку»; MCP до кнопки не вызван.

Это проверка двух сценариев, а не полный обязательный эксперимент на 12 запросах.
Работающий пользовательский бот автоматически не перезапускался. Для применения
нового кода нужен перезапуск; schema initialization выполнится при старте.

## /timezone по названию города (17.11.2026)

Пользователь уточнил, что нужна общая поддержка городов, а не alias одной Москвы.
Добавлен TimezoneResolver: валидная IANA-зона работает локально; для города используется
публичный Open-Meteo Geocoding. Проверены timezone/name/country/admin1/population,
тайм-ауты, отказ от redirects, ограничение ответа и обработка ошибки без изменения БД.

Точные имена приоритетны. Одинаковые IANA-ключи позволяют выбрать пояс без различения
мест; при разных ключах явно крупный город может быть выбран по населению. Это
предположение приложения: handler показывает город и предлагает поправить страну/регион.
При близких лидерах, ничьей или недостаточных данных нужна уточнённая команда;
прежняя настройка сохраняется. Разные IANA-aliases консервативно считаются разными ключами.

Проверки:

- `python -B -m pytest -q tests/test_timezones.py tests/test_reminder_handlers.py
  tests/test_reminder_agent.py tests/test_lifecycle.py`: **62 passed**, 12,50 s.
- Реальный API: Москва → Europe/Moscow; Казань → Europe/Moscow;
  Ереван → Asia/Yerevan; Токио → Asia/Tokyo;
  «Париж, Франция» → Europe/Paris. Все пять PASS, без Telegram и записи в рабочую БД.
- Ruff check/format и git diff --check: PASS. Независимое ревью: блокеров нет.
- Полный pytest и PostgreSQL-интеграция для этого дополнения повторно не запускались:
  схема и ReminderStore не менялись. Предыдущие результаты приведены выше.
