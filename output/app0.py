from dash import Dash, dcc, html, Input, Output, dash_table, callback
import polars as pl
import json

news = True

if news:
    data = pl.read_csv('news_clean.csv', 
                    dtypes={
                            "content_hash": pl.UInt64,
                            "date_extract": pl.Date,
                            "date_article": pl.Date
                        }
                        )

    main_table = data[['state', 'city', 'date_article', 'sumary', 'link' ]]
    df = main_table.to_pandas()
    df['id'] = df['link']
    df.set_index('id', inplace=True, drop=False)
else:
    data = pl.read_csv('ner.csv', 
                    dtypes={
                    #        "content_hash": pl.UInt64,
                    #        "date_extract": pl.Date,
                    #        "date_article": pl.Date
                    'index': pl.UInt8,
                        }
                        )   
    df = data.to_pandas()
    df['id'] = df['index']
    df.set_index('id', inplace=True, drop=False)

app = Dash(__name__)

app.layout = html.Div([

    dcc.RadioItems(
        [{'label': 'Read filter_query', 'value': 'read'}, {'label': 'Write to filter_query', 'value': 'write'}],
        'read',
        id='filter-query-read-write',
    ),

    html.Br(),

    dcc.Input(id='filter-query-input', placeholder='Enter filter query'),

    html.Div(id='filter-query-output'),

    html.Hr(),

    dash_table.DataTable(
        id='datatable-advanced-filtering',
        columns=[
            {'name': i, 'id': i, 'deletable': True} for i in df.columns
            # omit the id column
            if i != 'id'
        ],
        data=df.to_dict('records'),
        editable=True,
        page_action='native',
        style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(220, 220, 220)',
        }
        ],
        style_cell={
            'textAlign': 'left',
            'fontSize':28, 
            'font-family':'sans-serif'
            },
        style_header={
        'backgroundColor': 'gray',
        'fontWeight': 'bold'
        },
        style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
        },
        page_size=100,
        filter_action="native"
    ),
    html.Hr(),
    html.Div(id='datatable-query-structure', style={'whitespace': 'pre'})
])


@callback(
    Output('filter-query-input', 'style'),
    Output('filter-query-output', 'style'),
    Input('filter-query-read-write', 'value')
)
def query_input_output(val):
    input_style = {'width': '100%'}
    output_style = {}
    if val == 'read':
        input_style.update(display='none')
        output_style.update(display='inline-block')
    else:
        input_style.update(display='inline-block')
        output_style.update(display='none')
    return input_style, output_style


@callback(
    Output('datatable-advanced-filtering', 'filter_query'),
    Input('filter-query-input', 'value')
)
def write_query(query):
    if query is None:
        return ''
    return query


@callback(
    Output('filter-query-output', 'children'),
    Input('datatable-advanced-filtering', 'filter_query')
)
def read_query(query):
    if query is None:
        return "No filter query"
    return dcc.Markdown('`filter_query = "{}"`'.format(query))


@callback(
    Output('datatable-query-structure', 'children'),
    Input('datatable-advanced-filtering', 'derived_filter_query_structure')
)
def display_query(query):
    if query is None:
        return ''
    return html.Details([
        html.Summary('Derived filter query structure'),
        html.Div(dcc.Markdown('''```json
{}
```'''.format(json.dumps(query, indent=4))))
    ])


if __name__ == '__main__':
    app.run(debug=True)
