# Activación_HPGe

## Activacion.py

Contiene funciones útiles para el análisis de espectros en formato .txt (usando _export_ del software GammaVision, de ORTEC). Devuelve los datos que se analizan a mano en un software de un multicanal. Además, sirve para las calibraciones en eficiencia (fuentes patrones e incógnitas), cálculos de flujo y actividad (definición del objeto Alambre para analizar los espectros de las hojuelas irradiadas), descarga de datos del Livechart de la IAEA usando su API, ajuste de datos y reporte de parámetros de bondad. 

## NAA_HPGe
Esquema general de programa que implementa las funciones de Activación.py en un caso particular. En simples pasos, es posible: 
- Calibrar en eficiencia el detector de germanio a partir de sus espectros (fondos y fuentes). Para esto se define un guess inicial usando un patrón, y luego se agregan puntos por fuentes incógnitas. También se compara la actividad reportada de la calculada a partir del guess inicial. 
- Definición de hojuelas/alambres, y cálculo de sus tasas, actividades y valores de flujo neutrónico al final de la irradiación.
- Devolución de reportes y gráficos en cada paso que se especifique.
- Como la bondad del ajuste y los parámetros estimados para las calibraciones son los datos más importantes, son los únicos que se diferencian al ir actualizando el array de valores de eficiencia.  
