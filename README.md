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
