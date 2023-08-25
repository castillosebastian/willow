import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from IPython.display import Markdown
from tabulate import tabulate
import networkx as nx
import matplotlib.pyplot as plt


def consolidate_news(news):
    try:
        # Removing duplicates based on 'content_hash'
        news = news.unique(subset=['content_hash'])
        news = news.unique(subset=['link'])
    except Exception as e:
        raise Exception(f"Error in consolidation step: {str(e)}")
    
    return news

def consolidate_ner(newsner):
   
    try:
        # elimino signos de puntuación        
        newsner = (
            newsner.with_columns(
                pl.col('word').str.replace_all(r'[^\w\s]', ' ').str.strip().alias('word')
            )
        )
        # elimino blancos
        newsner = (
            newsner
            .filter((pl.col('word') != ''))                    
        )
        # elimino new en blanco
        #newsner = (
        #    newsner
        #    .filter(pl.col('word').str.lengths()>0)                    
        #)
        
        # una entrada por artículo. Ojo me puedo estar quedando con score bajo en la entrada
        newsner = newsner.unique(subset=['word', 'index'])

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
                    'index', 'state', 'date_article', 'portal', 'authors', 'title', 'summary_llm', 'link'
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
    table['hyperlink'] = table['link'].apply(lambda x: f'\\href{{{x}}}{{link}}') 
    
    return table.reset_index(drop=True)


def table_ner(newsner):

    try:
        newsner = newsner.to_pandas()

        newsner = (
            newsner.groupby(['index', 'entity_group'])['word']
            .apply(lambda x: '; '.join(x))
            .reset_index()
        )

        newsner = newsner.pivot(index='index', 
                                columns='entity_group', 
                                values='word').reset_index()
        
         # Drop the 'entity_group' column
        #newsner = newsner.drop(columns=['entity_group'])

        # Rename the columns
        newsner = newsner.rename(columns={
            'index': 'Ref.art',
            'LOC': 'lugares',
            'MISC': 'varios',
            'ORG': 'organizaciones',
            'PER': 'personas'
        })

        # Reorder the columns
        newsner = newsner[['Ref.art', 'personas', 'lugares', 'organizaciones', 'varios']]

    except Exception as e:
            raise Exception(f"Error in table  ner: {str(e)}")
    
    #newsner = newsner.drop(columns=['entity_group'])
    
    return newsner


def ner_to_network(newsner):
    try:
        # Converting to pandas DataFrame
        df = newsner.to_pandas()

        # Selecting relevant columns and renaming
        df = df[['word', 'index', 'entity_group']]
        df.rename(columns={'index': 'article'}, inplace=True)

        # Filtering based on entity group and non-numeric words
        df = df[
            (df['entity_group'] == 'PER') &
            (~df['word'].str.isnumeric())
        ]

        # Filtering to retain values with two or more words (by counting spaces)
        df = df[df['word'].str.count(' ') >= 1]  # 1 space means 2 words, First and Second name patern

        # Grouping by 'word' and aggregating 'article' into a list
        transformed_df = df.groupby('word')['article'].apply(list).reset_index(name='articles')
        transformed_df['occurrences'] = transformed_df['articles'].apply(len)

        return transformed_df

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def draw_top_ner(df, top_n):
    # Create a figure object
    fig = plt.figure(figsize=(10, 15)) # You can adjust the width and height as needed

    # Create a Graph
    G = nx.Graph()

    # Select the top N words based on article occurrences
    top_words = df.nlargest(top_n, 'occurrences')

    # Add nodes and edges for words and articles
    for _, row in top_words.iterrows():
        word_node = row['word']
        G.add_node(word_node, type="word")
        for article_index in row['articles']:
            article_node = f"{article_index}"
            G.add_node(article_node, type="article")
            G.add_edge(word_node, article_node)

    # Draw nodes and edges using a spring layout
    pos = nx.spring_layout(G, k=0.3)

    # Draw word nodes
    word_nodes = [node for node in G.nodes if G.nodes[node]['type'] == 'word']
    nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_color='blue', node_size=500)

    # Draw article nodes
    article_nodes = [node for node in G.nodes if G.nodes[node]['type'] == 'article']
    nx.draw_networkx_nodes(G, pos, nodelist=article_nodes, node_color='gray', node_size=200)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Draw labels for words and articles
    labels = {node: node for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)

    # Add legends for node types
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Entidades'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Artículos')],
               loc='best')

    # Add a title
    plt.title("Entidades más nombradas entre los artículos analizados")

    # Return the figure object
    return fig
    
