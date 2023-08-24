import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from IPython.display import Markdown
from tabulate import tabulate


def consolidate_news(news):
    try:
        # Removing duplicates based on 'content_hash'
        news = news.unique(subset=['content_hash'])
    except Exception as e:
        raise Exception(f"Error in consolidation step: {str(e)}")
    
    return news

def consolidate_ner(newsner):
    try:        
        newsner = (
            newsner.with_columns(
                pl.col('word').str.replace_all(r'[^\w\s]', ' ').str.strip().alias('word')
            )
        )
        newsner = (
            newsner
            .filter((pl.col('word') != ''))                    
        )
    except Exception as e:
        raise Exception(f"Error in consolidation step: {str(e)}")    
    return newsner

def get_news_stat(news, newsner):
    try:
        # Step 1: Transformation on 'news' dataframe
        noticias_xprov = (
            news.filter(pl.col("state") != "Argentina")
            .groupby("state").agg(
                pl.count("state").alias("cantidad_noticias")
            )
        ).sort(by="cantidad_noticias", descending=True)
    except Exception as e:
        raise Exception(f"Error in Step 1: {str(e)}")

    try:
        # Step 2: Transformation on 'newsner' dataframe
        newsner = newsner.join(news.select(['index', 'state']), on='index', how='left')
        
        result = (
            newsner
            .filter((pl.col("state").is_not_null()) & (pl.col("state") != "Argentina"))
            .groupby(["state", "entity_group"])
            .agg(pl.count("entity_group").alias("count"))
        )
    except Exception as e:
        raise Exception(f"Error in Step 2: {str(e)}")

    try:
        # Step 3: Pivot result
        pivot_result = result.pivot(
            index="state",
            columns="entity_group",
            values="count",
            aggregate_function=pl.col("count").sum()
        )
        pivot_result = pivot_result.rename(
            {
                "MISC": "varios",
                "LOC": "lugares",
                "ORG": "organizaciones",
                "PER": "personas"
            }
        )
    except Exception as e:
        raise Exception(f"Error in Step 3: {str(e)}")

    try:
        # Step 4: Combine the dataframes
        newstat = noticias_xprov.join(pivot_result, on='state', how='left')
        newstat = newstat.select(
            [
                'state', 'cantidad_noticias', 'personas', 'lugares',
                'organizaciones', 'varios'
            ]
        ).rename({'state': 'provincia'})
        
        sumrow = newstat.sum()
        sumrow = sumrow.with_columns(pl.col("provincia").fill_null("Total"))
        newstat = pl.concat([newstat, sumrow])
        

    except Exception as e:
        raise Exception(f"Error in Step 4: {str(e)}")
    
    newstat = newstat.to_pandas()
    newstat.index = np.arange(1, len(newstat) + 1)

    return newstat

def plot_dataframe(df, x_col, y_col, x_label, y_label, title, color='skyblue', figsize=[10,6]):
    # Filtering out rows where 'provincia' is 'Total'    
    df = df[~df['provincia'].str.contains('Total', case=False)]

    plt.figure(figsize=figsize)
    plt.barh(df[y_col], df[x_col], color=color)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.show()


def table_news(news, type = 'abstract'):

    if type =='abstract':
        try:
            table = (
                news.with_columns(
                    pl.col("link").str.extract(r"www.(\w+)", 1).alias("portal"),
                    pl.when(pl.col("date_article").is_null())
                    .then(pl.col("date_extract"))
                    .otherwise(pl.col("date_article"))
                    .alias("date_article"),         
                )
            )

            table = (
                table.select(
                [
                    'index', 'state', 'date_article', 'portal', 'authors', 'title', 'summary_llm'
                ]
                )
                .sort(["state", "date_article"])
                
            )

            table = (
                table.with_columns(
                    pl.when(pl.col("portal") == 'argentina')
                    .then(pl.lit('Gendarmería'))
                    .otherwise(pl.col("portal"))
                    .alias("portal"),           
                )
            )

            table = (
                table.with_columns(
                    pl.when(pl.col("portal").is_null())
                    .then(pl.lit('n-a'))
                    .otherwise(pl.col("portal"))
                    .alias("portal"),           
                )
            )

            table = table.rename({
                'index': 'Ref.art',
                'state': 'Pcia_Estado',
                'date_article': 'fecha_art',
                'authors': 'autores',
                'title': 'titular', 
                'summary_llm': 'resumen'
                })
            
        except Exception as e:
            raise Exception(f"Error in table news abstract: {str(e)}")
    
    table = table.to_pandas()    
    table['fecha_art'] = table['fecha_art'].dt.strftime('%Y-%m-%d')

    return table.reset_index(drop=True)