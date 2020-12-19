print(0.1+0.1+0.1==0.3)
# Esto es por que en base binaria
# 0.1 (decimal) = 0.0001100110011001100110011...(binario)
# La maquina trabaja estos numeros en (binario) en este caso es periodico

"""Notemos las diferencias"""
# Round(x,n) : Aproximacion por redondeo de x con n decimales
round(.25,1) # 0.2
round(.26,1) # 0.3

# Si primero redondeamos y luego sumo
# Tenemos exactamente la exprecion de arriba
print(round(.1,1)+round(.1,1)+round(.1,1) == round(.3,1))
# Si primero sumo y luego redondeamos
# 0.1 + 0.1 + 0.1 => 0.30000000000000004
print(round(.1+.1+.1,10) == round(.3,10))


# Format(x,e) : Ver como representa python un numero flotante, con una precicion de 'e'
print(format(0.1,'.17f')) #Aca Fijate como se guarda en python
print(format(0.3,'.17f')) #Aca Tambien, Ahora tiene sentido
# Fijarse por que al redondearse ocurre el problema


x = 3.14159
print(x.as_integer_ratio())
# Devuelve su representacion (p/q) donde p y q pertenecen a enteros

