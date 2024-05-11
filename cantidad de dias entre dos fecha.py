from datetime import datetime

# Definir la fecha antigua
fecha_antigua = datetime(2023, 1, 1)  # Por ejemplo, el 1 de enero de 2023

# Obtener la fecha actual
fecha_actual = datetime.now()

# Calcular la diferencia de días entre la fecha actual y la fecha antigua
diferencia = fecha_actual - fecha_antigua

# Extraer el número de días de la diferencia
cantidad_dias = diferencia.days

# Mostrar la cantidad de días
print("Cantidad de días entre hoy y la fecha antigua:", cantidad_dias)
