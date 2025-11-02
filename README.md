# NYC Taxi Data Analysis with Apache Spark

Цей проект використовує Apache Spark для завантаження та аналізу CSV файлів з даними про поїздки таксі (yellow_tripdata). Дані завантажуються та обробляються в розподіленому Spark кластері.

## Архітектура

Проект складається з трьох Docker контейнерів:

- **spark-master** - головний вузол Spark кластера
- **spark-worker** - робочий вузол для обробки даних
- **spark-app** - застосунок Python, який запускає Spark job

## Вимоги

- Docker
- Docker Compose
- Python 3.9+

## Швидкий старт

### 1. Клонуйте репозиторій

```bash
git clone <repository-url>
cd taxi-spark-project
```

### 2. Підготовка даних

Помістіть ваші CSV файли в директорію `data/`:

```
data/
  └── yellow_tripdata_2015-01.csv
```

### 3. Запуск проекту

```bash
docker-compose up --build
```

### 4. Перегляд результатів

Відкрийте браузер та перейдіть до:
- **Spark Master UI**: http://localhost:8080

## Команди

### Запуск всіх сервісів

```bash
docker-compose up
```

### Запуск у фоновому режимі

```bash
docker-compose up -d
```

### Зупинка всіх контейнерів

```bash
docker-compose down
```

### Видалення всіх контейнерів та volumes

```bash
docker-compose down -v
```

## Опис датасету

**Джерело:** [NYC Yellow Taxi Trip Data (Kaggle)](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data/data)

**Період:** Січень 2015  
**Кількість записів:** 12,748,986  
**Кількість колонок:** 19

### Структура даних

| Поле | Тип даних | Nullable | Опис |
|------|-----------|----------|------|
| `vendor_id` | String | ✓ | Ідентифікатор постачальника послуг таксі (1 = Creative Mobile Technologies, 2 = VeriFone Inc.) |
| `tpep_pickup_datetime` | Timestamp | ✓ | Дата та час початку поїздки |
| `tpep_dropoff_datetime` | Timestamp | ✓ | Дата та час завершення поїздки |
| `passenger_count` | Integer | ✓ | Кількість пасажирів у транспортному засобі |
| `trip_distance` | Double | ✓ | Відстань поїздки в милях |
| `pickup_longitude` | Double | ✓ | Довгота місця посадки пасажира |
| `pickup_latitude` | Double | ✓ | Широта місця посадки пасажира |
| `rate_code` | Integer | ✓ | Тарифний код (1 = Standard rate, 2 = JFK, 3 = Newark, 4 = Nassau/Westchester, 5 = Negotiated fare, 6 = Group ride) |
| `store_and_fwd_flag` | String | ✓ | Прапорець збереження поїздки в пам'яті транспортного засобу (Y = так, N = ні) |
| `dropoff_longitude` | Double | ✓ | Довгота місця висадки пасажира |
| `dropoff_latitude` | Double | ✓ | Широта місця висадки пасажира |
| `payment_type` | String | ✓ | Спосіб оплати (1 = Credit card, 2 = Cash, 3 = No charge, 4 = Dispute, 5 = Unknown, 6 = Voided trip) |
| `fare_amount` | Double | ✓ | Базова вартість поїздки за тарифом |
| `extra` | Double | ✓ | Додаткові збори (нічний тариф, години пік) |
| `mta_tax` | Double | ✓ | Податок MTA (0.50 USD) |
| `tip_amount` | Double | ✓ | Чайові (автоматично для карткових платежів, готівкові чайові не включені) |
| `tolls_amount` | Double | ✓ | Сума оплати за проїзд платними дорогами |
| `improvement_surcharge` | Double | ✓ | Доплата на покращення (0.30 USD) |
| `total_amount` | Double | ✓ | Загальна сума оплати (без готівкових чайових) |

### Про nullable значення

Усі поля в датасеті мають `nullable=True`, що означає можливість відсутності значень (NULL).

**Чому це важливо:**
- Реальні дані містять пропуски через технічні збої, помилки введення або неповну фіксацію інформації
- В даному датасеті виявлено **3 пропуски** в полі `improvement_surcharge`
- Координати можуть бути NULL, якщо GPS не зафіксував дані
- `tip_amount` може бути NULL або 0 для готівкових платежів (система не фіксує готівкові чайові)

**Рекомендації з обробки:**
```python
from pyspark.sql.functions import col, count, when

df.select([
    count(when(col(c).isNull(), c)).alias(c) 
    for c in df.columns
]).show()

df_clean = df.dropna()
```

## Аналіз та висновки

### Загальна інформація про набір даних

Використовується набір даних **NYC Taxi Trip Data (January 2015)**.

- Кількість рядків: **12 748 986**
- Кількість колонок: **19**

Типи даних:
- **Часові:** `tpep_pickup_datetime`, `tpep_dropoff_datetime`
- **Числові:** `passenger_count`, `trip_distance`, `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount`
- **Географічні:** `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`
- **Категоріальні:** `vendor_id`, `rate_code`, `store_and_fwd_flag`, `payment_type`

### Пропуски в даних

Було виявлено лише **3 пропуски** у стовпці `improvement_surcharge`.  
Після видалення залишилось **12 748 983** рядки.

---

### Аналіз та очищення числових стовпців

#### Passenger Count
- Більшість поїздок: **1–2 пасажири**

#### Trip Distance
Виявлено аномальні значення (до мільйонів миль).

Рекомендована фільтрація:
```python
df = df.filter((df.trip_distance > 0) & (df.trip_distance < 100))
```

#### Географічні координати
Присутні значення, що виходять далеко за межі Нью-Йорка (наприклад, широта 0 або довгота 400).

Рекомендована фільтрація координат у межах міста:
```python
df = df.filter(
    (df.pickup_latitude.between(40.5, 41.0)) &
    (df.pickup_longitude.between(-74.3, -73.6)) &
    (df.dropoff_latitude.between(40.5, 41.0)) &
    (df.dropoff_longitude.between(-74.3, -73.6))
)
```
### Fare Amount

Виявлено негативні та нереалістично великі значення.

```
df = df.filter((df.fare_amount > 0) & (df.fare_amount < 1000))
```

Tip Amount

Присутні аномально великі значення (до мільйонів доларів).
```
df = df.filter((df.tip_amount >= 0) & (df.tip_amount < 100))
```

Tolls Amount

Більшість значень 0, але трапляються аномалії.
```
df = df.filter((df.tolls_amount >= 0) & (df.tolls_amount < 100))
```
Improvement Surcharge

Стовпець коректний, очікуване значення 0.3, значних відхилень немає.

Підсумкові висновки

Дані практично не містять пропусків, однак мають аномальні значення у відстанях, координатах та платіжних сумах.

Перед подальшим аналізом або побудовою моделей виконано очищення та фільтрацію.

###
