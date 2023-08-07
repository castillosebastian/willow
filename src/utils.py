import polars as pl
import os 

keywords = [
    'narcotráfico',
    'drogas',
    'cocaína',
    'marihuana',
    'heroína',
    'anfetaminas',
    'metanfetaminas',
    'éxtasis',
    'crimen organizado',
    'traficante',
    'cartel',
    'narcos',
    'estupefacientes',
    'psicotrópicos',
    'incautación',
    'tráfico de drogas',
    'dealer',
    'mafia',
    'pasta base',
    'crack',
    'opiáceos',
    'fentanilo',
    'alcaloide',
    'sintéticas',
    'laboratorio clandestino',
    'lavado de dinero',
    'blanqueo de capitales',
    'corrupción'
]

def load_urls():
    df = pl.read_csv('data/portals.csv')
    return df['newsportalurl'].to_list()
