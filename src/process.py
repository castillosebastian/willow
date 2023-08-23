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
        
    except Exception as e:
        raise Exception(f"Error in Step 4: {str(e)}")
    
    return news, newsner, newstat

def plot_dataframe(df, x_col, y_col, x_label, y_label, title, color='skyblue', figsize=[10,6]):
    plt.figure(figsize=figsize)
    plt.barh(df[y_col].to_list(), df[x_col].to_list(), color=color)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.show()
