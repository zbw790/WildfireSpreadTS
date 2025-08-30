
# WildfireSpreadTS 全面数据质量分析报告

## 数据概览
- **总样本数**: 337,664,323
- **总文件数**: 200
- **独特火灾事件**: 200
- **特征数**: 23

## NaN值问题分析

### 严重NaN问题 (>5%):
无严重NaN问题

### 中等NaN问题 (1-5%):
- **VIIRS_I4**: 2.1% NaN值
- **VIIRS_I5**: 2.1% NaN值
- **VIIRS_M13**: 2.1% NaN值
- **NDVI**: 1.7% NaN值
- **EVI2**: 1.7% NaN值
- **PDSI**: 1.3% NaN值

### 轻微NaN问题 (<1%):
- **Temperature**: 1.0% NaN值
- **Humidity**: 1.0% NaN值
- **Wind_Direction**: 1.0% NaN值
- **Wind_Speed**: 1.0% NaN值
- **Precipitation**: 1.0% NaN值
- **Surface_Pressure**: 1.0% NaN值
- **Solar_Radiation**: 1.0% NaN值
- **Elevation**: 0.5% NaN值
- **Slope**: 0.5% NaN值
- **Aspect**: 0.5% NaN值

## NaN值处理建议

### 推荐策略:
1. **严重NaN通道**: 考虑特征工程或删除
2. **中等NaN通道**: 使用插值或基于邻近像素的方法
3. **轻微NaN通道**: 简单均值或中位数插值

### 针对CNN训练的具体建议:
- 使用 `sklearn.impute.SimpleImputer` 进行中位数插值
- 对于土地覆盖等分类特征，使用最频繁值填充
- 考虑添加NaN指示特征来保留缺失信息

## 数据范围分析
- **VIIRS_I4**: [-100.00, 16000.00]
- **VIIRS_I5**: [-100.00, 15998.00]
- **VIIRS_M13**: [-100.00, 15987.00]
- **NDVI**: [-9951.00, 9995.00]
- **EVI2**: [-5163.00, 9998.00]
- **Temperature**: [0.00, 145.30]
- **Humidity**: [0.40, 14.70]
- **Wind_Direction**: [0.00, 357.00]
- **Wind_Speed**: [245.70, 311.50]
- **Precipitation**: [254.70, 323.20]
- **Surface_Pressure**: [0.00, 117.00]
- **Solar_Radiation**: [0.00, 0.02]
- **Elevation**: [0.00, 67.07]
- **Slope**: [-0.00, 359.89]
- **Aspect**: [-45.00, 4350.00]
- **PDSI**: [-8.57, 7.91]
- **Land_Cover**: [1.00, 17.00]
- **Forecast_Temperature**: [0.00, 1144.81]
- **Forecast_Humidity**: [0.00, 10.07]
- **Forecast_Wind_Direction**: [-89.99, 90.00]
- **Forecast_Wind_Speed**: [-17.04, 37.20]
- **Forecast_Precipitation**: [0.00, 0.01]
- **Active_Fire_Confidence**: [0.00, 22.00]
