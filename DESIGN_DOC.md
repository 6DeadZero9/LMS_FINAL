# Фінальне ТЗ: Проектування сенсорної системи

**Студент:** Ivan Iziumov  
**Дата:** 25.09.2026

---

## 1. Use Case та постановка задачі

**Use case:** indoor mobile robot (TurtleBot3 Burger) — лабораторна платформа для оцінки сенсорної фільтрації пози на плоскій підлозі.

**Сценарій роботи:** Робот у симуляторі Gazebo стартує в `(0, 0)` і один раз проїжджає квадратний маршрут 2×2 м між чотирма відомими віхами (стовпами), після чого зупиняється. Паралельно EKF оцінює позу за IMU, колісною одометрією та LiDAR; GUI порівнює чотири комбінації сенсорів з ground truth Gazebo (ATE / yaw RMSE).

---

## 2. Вимоги до системи

| Параметр | Значення | Обґрунтування |
|----------|----------|---------------|
| Точність позиції | ≤ 0.10 m (ціль для IMU+odom+LiDAR); IMU-only може деградувати до >0.5 m | Достатньо для утримання в коридорі квадрата 2 м; показує виграш ф’южну vs дрейф IMU |
| Частота оновлення стану | ~20 Hz оцінка / публікація; IMU predict ~200 Hz | DiffDrive odom ~30 Hz, LiDAR ~5 Hz; 20 Hz досить для pursuit і графіків |
| Затримка (latency) | < 100 ms end-to-end (sim time) | Онлайн у симуляції; не запис offline |
| Операційний температурний діапазон | 0…+40 °C | Indoor lab / офіс |
| Споживання | ~15–25 Вт середньо (платформа TB3) | Бортова електроніка + LiDAR + мотори; у проєкті — симуляція |
| Бюджет на сенсори | ~$250–400 (типовий TB3 Burger kit) | IMU на платі керування + LDS-01 + енкодери коліс |
| Вологозахист / IP-рейтинг | IP20 (indoor dry) | Немає вимог до вулиці / бризок |

---

## 3. Sensor Allocation Table

| # | Сенсор | Модель | Призначення | Інтерфейс | Частота (Hz) | Вартість ($) |
|---|--------|--------|-------------|-----------|--------------|--------------|
| 1 | IMU | Gazebo IMU plugin (аналог MPU на OpenCR TB3) | \(a_x\), \(\omega_z\) для predict EKF; yaw bias | Внутрішній / bridged ROS `/imu` | ~200 | ~10–20 (на платі) |
| 2 | Wheel odometry | DiffDrive + енкодери | Body linear speed \(v\) (не поза) | ROS `/odom` twist | ~30 | ~0 (входить у привід) |
| 3 | 2D LiDAR | LDS-01 (HLS-LFCD) | Range/bearing до 4 відомих стовпів | USB / ROS `/scan` | ~5 | ~90–120 |
| 4 | Ground truth | Gazebo entity pose `burger` | Еталон для ATE/yaw і для path following | Gazebo Transport → `/ground_truth/odom` | ~sim rate | $0 (симуляція) |

**Обґрунтування вибору кожного сенсора:**

1. **IMU:** базова частота для unicycle predict; на TB3 уже є; у симуляторі додаємо bias/шум у ПЗ, щоб IMU-only дрейфував.
2. **Wheel odometry:** дешева корекція швидкості; у фільтрі беремо лише twist \(v\), щоб не «затискати» коваріацію й не витісняти LiDAR.
3. **LiDAR LDS-01:** стандарт Burger; полюси з відомими координатами дають абсолютні range/bearing оновлення.
4. **Gazebo GT:** Vicon-аналог у симі; потрібен для RMSE і щоб рух не залежав від якості оцінки.

---

## 4. Block Diagram

```
                 [ Host PC: ROS 2 + Gazebo (ros-gz) ]
                              |
            +-----------------+------------------+
            |                 |                  |
     [Gazebo world]    [ros_gz_bridge]    [sensor_engineering]
     burger+poles      /imu /odom /scan   nodes + EKF + GUI
     DiffDrive         /cmd_vel /clock
            |                 |                  |
            |                 v                  v
            |          [fusion_node]      [path_follower]
            |          4× EKF suites      pure pursuit (GT pose)
            |                 |                  |
            |                 v                  v
            |          [fusion_dashboard]   /cmd_vel → DiffDrive
            |          ATE / yaw plots
            v
     Gazebo Transport pose → [ground_truth_node]
```

Дані сенсорів йдуть у `fusion_node`; керування йде з `path_follower` за GT; GUI лише вибирає suite і показує помилки.

---

## 5. Power Budget

*(Орієнтовно для реального TB3 Burger у indoor-режимі; у курсовому проєкті платформа симулюється.)*

| Компонент | Струм (мА) | Duty Cycle | мА·год/добу |
|-----------|-----------|------------|-------------|
| SBC / MCU (active, Pi-клас) | 400 | 0.3 (рух/обчислення) | 2880 |
| MCU/SBC (idle) | 150 | 0.7 | 2520 |
| IMU (на платі) | 5 | 1.0 | 120 |
| LiDAR LDS-01 | 300 | 0.5 (під час місії) | 3600 |
| Мотори + драйвер (середнє) | 800 | 0.25 | 4800 |
| WiFi / ROS link | 100 | 0.5 | 1200 |
| **TOTAL** | | | **~15120** |

**Розрахунковий час роботи від 1800 мА·год (типовий пак TB3 ~11.1 V):** при середньому ~630 мА → близько **2.5–3 год** безперервної місії (не «днів»; мобільний робот, не wearables).

---

## 6. FMEA (Failure Mode and Effects Analysis)

| Сенсор | Режим відмови | Ефект на систему | Severity (1–5) | Mitigation |
|--------|---------------|------------------|----------------|------------|
| IMU | Stuck / bias jump | Дрейф yaw і пози в IMU-only / без LiDAR | 4 | Оцінка \(b_g\) в EKF; LiDAR suite; synthetic bias лише в ПЗ для демо |
| Wheel odom | Slip / хибний \(v\) | Зсув швидкості, гірший predict | 3 | Великий \(R_v\); LiDAR absolute updates; не оновлювати yaw з коліс |
| LiDAR | Втрата стовпа / хибна асоціація | Помилковий update пози | 4 | Mahalanobis gate; асоціація за позою фільтра; 4 redundancy poles |
| Bridge / QoS | Немає `/imu` чи `/scan` | Порожні графіки / застигла оцінка | 5 | Sensor-data QoS; перевірка топіків після launch |
| Gazebo / spawn | Порожня сцена, немає burger | Немає GT і руху | 5 | Burger у SDF; один процес `gz sim`; `pkill` старих серверів |
| Power | Battery undervoltage | Системний reset | 5 | Voltage monitoring + graceful shutdown |

---

## 7. Data Source для курсового проекту

**Обраний data source для симуляції/реалізації:**
- [ ] Власний генератор шуму (lib/noise_models.py, конфігурований ARW/bias)
- [ ] EuRoC MAV dataset (реальний IMU + Vicon ground truth)
- [ ] KITTI odometry (реальний LiDAR + GPS)
- [ ] nuScenes (radar + LiDAR + camera)
- [x] Webots (3D-симулятор: наземний / дрон / надводний світ) — **у реалізації: Gazebo Sim (ros-gz) + TurtleBot3**, той самий клас (повний 3D-симулятор з сенсорами й GT)

**Обґрунтування вибору:** Потрібен контрольований indoor сценарій з відомими віхами, синхронними `/imu`, `/odom`, `/scan` і точним pose GT. Gazebo/TurtleBot3 дає це з ROS 2 out of the box; шум/bias IMU додатково задаються в `fusion.yaml` для порівняння suite.

---

## 8. Planned Pipeline

```
[ Gazebo TB3 ] --> [ ros_gz_bridge + noise/bias ] --> [ EKF ×4 suites ] --> [ /fusion/<mode>/odom ]
        |                                                      |
        +--> [ GT pose ] --> [ pure pursuit ] --> [ /cmd_vel ]
        |                                                      |
        +--> [ dashboard: RMSE / ATE vs GT, suite select ]
```

Gazebo генерує сенсори й істинну позу. Bridge віддає їх у ROS; `fusion_node` робить predict з IMU та updates з odom/LiDAR у чотирьох режимах. `path_follower` веде квадрат за GT (оцінка не керує моторами). Dashboard показує траєкторії й RMSE та перемикає активний suite через `/fusion/selection`.

