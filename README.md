# ML_business_project
Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy API: flask Данные: с kaggle - Hotel Reservations Dataset (https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)

Задача: Нужно предсказть будет ли подтверждено бронирование или клиент откажеться от брони (поле booking_status). Бинарная классификация

Используемые признаки:

type_of_meal_plan - Тип плана питания забронированного клиентом (категории - Meal Plan 1, Meal Plan 2, Meal Plan 3, Not Selected)

required_car_parking_space - Признак отражающий необходимость парковочного места для клиента (int)

lead_time - Количество дней с момента бронирования до момента заезда в отель (int)

arrival_month - Месяц прибытия в отель (int)

market_segment_type - Тип сегмента рынка (категория - Online, Offline, Corporate, Complementary, Aviation)

repeated_guest - являеться ли данный визит повторным (int)

avg_price_per_room - средняя цена за номер. Цена за номер может меняться динамечиски (float)

no_of_special_requests - Общее количество особых запросов сделанных клиентом (int)

no_of_people - Количество людей (float)

no_of_week_days - Количество дней которые клиент проживает или бронирует для проживания в отеле (float)

booking_status - Признак того было отменено бронирование или нет

Преобразования признаков: RobustScaler, OHE

Модель: RandomForrest

Клонируем репозиторий и создаем образ

$ git clone https://github.com/vfeldchun/ML_for_business_project.git

$ cd ML_for_business_project

$ docker build -t vfeldchun/ml_business_docker .

Запускаем контейнер

Здесь Вам нужно создать каталог локально и сохранить туда предобученную модель (<your_local_path_to_pretrained_models> нужно заменить на полный путь к этому каталогу)

$ docker run -d -p 8180:8180 -p 8181:8181 -v <your_local_path_to_pretrained_models>:/app/app/models fimochka/vfeldchun/ml_business_docker

Переходим на localhost:8181
