# freMTPL2 数据集说明 / Data Dictionary

法国机动车第三者责任险（MTPL）数据，常用于精算定价和索赔建模研究。

---

## 一、数据格式概览

| 数据集 | 行数 | 列数 | 用途 |
|--------|------|------|------|
| **freMTPL2freq** | ~678,000 | 12 | 保单级：风险特征 + 索赔次数 |
| **freMTPL2sev** | ~26,000 | 2 | 索赔级：索赔金额 |

- **freq**：每条记录 = 一张保单（policy），包含该保单在观察期内的索赔次数
- **sev**：每条记录 = 一次索赔，通过 `IDpol` 与 freq 关联

---

## 二、freMTPL2freq 字段说明

| 字段 | 类型 | 说明 | 示例 / 取值范围 |
|------|------|------|-----------------|
| **IDpol** | integer | 保单唯一标识，用于关联 freMTPL2sev | 1, 3, 5, 10... |
| **ClaimNb** | integer | 观察期内的索赔次数（目标变量） | 0, 1, 2, 3, 4... |
| **Exposure** | numeric | 保单有效期限，单位：年 | 0.1 ~ 1（多数为 1） |
| **Area** | factor | 地区密度等级，A=农村 → F=城市中心 | A, B, C, D, E, F |
| **VehPower** | integer | 车辆功率（税务马力，有序） | 4 ~ 15 |
| **VehAge** | integer | 车辆年龄，单位：年 | 0 ~ 99 |
| **DrivAge** | integer | 驾驶人年龄，单位：年（法国 18 岁可驾驶） | 18 ~ 99 |
| **BonusMalus** | integer | 法国奖惩系数 | 50 ~ 350 |
| **VehBrand** | factor | 车辆品牌分组 | B1, B2, ..., B12 |
| **VehGas** | factor | 燃料类型 | Diesel, Regular |
| **Density** | integer | 驾驶人所在城市的人口密度，人/km² | 0 ~ 100,000+ |
| **Region** | factor | 保单所在大区（1970–2015 法国区划） | R11, R21, R22... |

---

## 三、关键字段详解

### 1. BonusMalus（奖惩系数）

法国 MTPL 的“无赔款优惠/有赔款加费”机制：
- **< 100**：bonus（有折扣）
- **= 100**：基准
- **> 100**：malus（加费）

最低常见为 50，最高可到 350 以上。

### 2. Area（地区等级）

按城市密度划分，A 最 rural，F 最 urban：
- A：农村
- B–E：低到高人口密度
- F：城市中心

### 3. ClaimNb 与 Exposure

- **ClaimNb**：该保单在 Exposure 年内的索赔次数（模型目标变量）
- **Exposure**：有效期限（年），多数 ≤ 1
- **频率**：`ClaimNb / Exposure`（每年平均索赔次数）

### 4. 连续型 vs 分类变量

| 连续型 | 分类型 |
|--------|--------|
| DrivAge, VehAge, VehPower, BonusMalus, Density | Area, VehBrand, VehGas, Region |

---

## 四、freMTPL2sev 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| **IDpol** | integer | 保单 ID，与 freMTPL2freq 关联 |
| **ClaimAmount** | numeric | 索赔金额（部分使用 IRSA-IDA 约定） |

---

## 五、notebook 中的常用处理

```r
# 1. 加载
dat <- read.csv("data/raw/freMTPL2freq.csv", header = TRUE)

# 2. 数据准备（与 Schelldorfer & Wüthrich 2019 一致）
dat$VehGas <- factor(dat$VehGas)
dat$ClaimNb <- pmin(dat$ClaimNb, 4)   # 截断 ≥4 的索赔
dat$Exposure <- pmin(dat$Exposure, 1) # 截断 >1 年的 exposure

# 3. 频率
dat$Frequency <- dat$ClaimNb / dat$Exposure
```

---

## 六、参考

- CASdatasets: https://dutangc.github.io/CASdatasets/reference/freMTPL.html
- Denuit et al. (2021), Insurance: Mathematics and Economics
- Noll et al. (2020), French motor third-party liability claims case study
