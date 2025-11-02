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
