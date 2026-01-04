# app.py
import base64
import io
from datetime import datetime

import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

app = Dash(__name__)
server = app.server

# ------------- Helpers -------------

def parse_contents(contents: str, filename: str) -> pd.DataFrame:
    """Decode Dash upload contents and return a DataFrame."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith('.csv'):
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8', errors='ignore')))
    elif filename.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(io.BytesIO(decoded))
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")

    # Basic cleaning / inference
    # Date inference: single-pass parsing with a pragmatic threshold.
    # We avoid errors='raise' because one malformed value would block recognition.
    for col in df.columns:
        if df[col].dtype == object:
            parsed = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            # Heuristic: treat as date if at least 70% parseable and has enough variation.
            if parsed.notna().mean() >= 0.7 and parsed.nunique(dropna=True) > 10:
                df[col] = parsed
    return df


def infer_columns(df: pd.DataFrame):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in numeric_cols + date_cols]
    return numeric_cols, date_cols, cat_cols


# ------------- Layout -------------
app.layout = html.Div([
    html.H1("DDDM Tool – Industry 4.0 Prototype", style={"marginBottom": 0}),
    html.P("Upload data, explore KPIs, and model scenarios interactively."),

    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag & Drop or ', html.A('Select Files (CSV/XLSX)')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '6px',
            'textAlign': 'center', 'margin': '10px 0'
        }, multiple=False
    ),

    dcc.Store(id='store-df'),

    html.Div(id='file-info', style={'margin': '6px 0', 'fontStyle': 'italic'}),

    html.Div([
        html.Div([
            html.Label('Select KPI (numeric)'),
            dcc.Dropdown(id='kpi-col', placeholder='Choose a numeric column…')
        ], style={'flex': 1, 'marginRight': '10px'}),
        html.Div([
            html.Label('Date column (optional)'),
            dcc.Dropdown(id='date-col', placeholder='If available, choose a date column…')
        ], style={'flex': 1, 'marginRight': '10px'}),
        html.Div([
            html.Label('Aggregation (for comparisons)'),
            dcc.Dropdown(
                id='agg-fn',
                options=[
                    {'label': 'Mean (average)', 'value': 'mean'},
                    {'label': 'Median', 'value': 'median'},
                    {'label': 'Sum', 'value': 'sum'},
                    {'label': 'Count (records)', 'value': 'count'},
                ],
                value='mean',
                clearable=False
            )
        ], style={'flex': 1})
    ], style={'display': 'flex'}),

    html.Hr(),

    html.Div([
        html.Div([
            html.H3('Preview'),
            dash_table.DataTable(
                id='table', page_size=10, style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '120px', 'width': '120px', 'maxWidth': '220px'}
            )
        ], style={'flex': 1, 'marginRight': '10px'}),
        html.Div([
            html.H3('Trend (if date column selected)'),
            dcc.Graph(id='trend-graph'),
            html.H3('Top Categories (auto chooses a categorical column)'),
            dcc.Graph(id='bar-graph')
        ], style={'flex': 1})
    ], style={'display': 'flex'}),

    html.Hr(),

    html.H2('Scenario Modeling'),
    html.P("Apply a multiplier to the selected KPI to simulate process changes (e.g., +10% production rate)."),
    html.Div([
        html.Div([
            html.Label('Multiplier (0.5x – 1.5x)'),
            dcc.Slider(id='multiplier', min=0.5, max=1.5, step=0.01, value=1.0,
                       tooltip={'placement': 'bottom', 'always_visible': True})
        ], style={'flex': 1, 'marginRight': '10px'}),
        html.Div([
            html.Label('Scenario Name'),
            dcc.Input(id='scenario-name', type='text', value='What‑if A', style={'width': '100%'})
        ], style={'flex': 1})
    ], style={'display': 'flex'}),

    dcc.Graph(id='scenario-graph'),

    html.Div([
        html.Button('Download Processed CSV', id='btn-download', n_clicks=0),
        dcc.Download(id='download-data')
    ], style={'margin': '10px 0'}),

    html.Div(id='footer', style={'color': '#777', 'fontSize': '12px', 'marginTop': '10px'})
])

# ------------- Callbacks -------------
@app.callback(
    Output('store-df', 'data'),
    Output('file-info', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def on_upload(contents, filename):
    if contents is None:
        return None, "No file uploaded yet."
    try:
        df = parse_contents(contents, filename)
        info = f"Loaded {filename} with {len(df):,} rows and {len(df.columns)} columns."
        return df.to_json(date_format='iso', orient='split'), info
    except Exception as e:
        return None, f"Error: {e}"


@app.callback(
    Output('kpi-col', 'options'),
    Output('kpi-col', 'value'),
    Output('date-col', 'options'),
    Output('date-col', 'value'),
    Input('store-df', 'data')
)
def populate_dropdowns(data_json):
    if not data_json:
        return [], None, [], None

    df = pd.read_json(data_json, orient='split')
    numeric_cols, date_cols, _ = infer_columns(df)

    kpi_opts = [{'label': c, 'value': c} for c in numeric_cols]
    date_opts = [{'label': c, 'value': c} for c in date_cols]

    default_kpi = numeric_cols[0] if numeric_cols else None
    default_date = date_cols[0] if date_cols else None

    return kpi_opts, default_kpi, date_opts, default_date


@app.callback(
    Output('table', 'data'), Output('table', 'columns'),
    Output('trend-graph', 'figure'), Output('bar-graph', 'figure'),
    Input('store-df', 'data'),
    Input('kpi-col', 'value'),
    Input('date-col', 'value'),
    Input('agg-fn', 'value')
)
def update_views(data_json, kpi_col, date_col, agg_fn):
    import plotly.graph_objects as go
    empty_fig = go.Figure()
    if not data_json:
        return [], [], empty_fig, empty_fig

    df = pd.read_json(data_json, orient='split')

    # DataTable preview
    columns = [{'name': c, 'id': c} for c in df.columns]
    data = df.head(200).to_dict('records')

    # Trend graph if date & KPI selected
    trend_fig = empty_fig
    if date_col and kpi_col and date_col in df.columns and kpi_col in df.columns:
        temp = df[[date_col, kpi_col]].dropna()
        temp = temp.sort_values(date_col)
        trend_fig = px.line(temp, x=date_col, y=kpi_col, title=f"Trend of {kpi_col} over time")

    # Bar graph: choose a categorical column with few unique values
    bar_fig = empty_fig
    cat_candidates = [c for c in df.columns if c not in [date_col, kpi_col]]
    cat_col = None
    for c in cat_candidates:
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 20:
            cat_col = c
            break
    if cat_col and kpi_col and kpi_col in df.columns:
        if agg_fn == 'count':
            agg = df.groupby(cat_col)[kpi_col].count().reset_index(name=kpi_col)
        elif agg_fn == 'sum':
            agg = df.groupby(cat_col)[kpi_col].sum().reset_index()
        elif agg_fn == 'median':
            agg = df.groupby(cat_col)[kpi_col].median().reset_index()
        else:
            agg = df.groupby(cat_col)[kpi_col].mean().reset_index().sort_values(kpi_col, ascending=False).head(10)
        bar_fig = px.bar(agg, x=cat_col, y=kpi_col, title=f"Top {cat_col} by {agg_fn} of {kpi_col}")

    return data, columns, trend_fig, bar_fig


@app.callback(
    Output('scenario-graph', 'figure'),
    Input('store-df', 'data'), Input('kpi-col', 'value'),
    Input('multiplier', 'value'), Input('scenario-name', 'value')
)
def scenario_model(data_json, kpi_col, multiplier, scen_name):
    import plotly.graph_objects as go
    fig = go.Figure()
    if not data_json or not kpi_col:
        fig.update_layout(title="Upload data and select a KPI to model a scenario")
        return fig

    df = pd.read_json(data_json, orient='split')
    if kpi_col not in df.columns:
        fig.update_layout(title="Selected KPI not found in dataset")
        return fig

    baseline = df[kpi_col].dropna()
    if baseline.empty:
        fig.update_layout(title="No data for selected KPI")
        return fig

    baseline_mean = baseline.mean()
    scenario_mean = baseline_mean * (multiplier or 1.0)

    fig.add_bar(name='Baseline', x=['Baseline'], y=[baseline_mean])
    fig.add_bar(name=scen_name or 'Scenario', x=['Scenario'], y=[scenario_mean])
    fig.update_layout(barmode='group', title=f"Scenario impact on {kpi_col} (mean)")
    return fig


@app.callback(
    Output('download-data', 'data'),
    Input('btn-download', 'n_clicks'),
    State('store-df', 'data'),
    prevent_initial_call=True
)
def download_processed(n, data_json):
    if not data_json:
        return None
    df = pd.read_json(data_json, orient='split')
    return dcc.send_data_frame(df.to_csv, filename=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)


@app.callback(
    Output('footer', 'children'),
    Input('store-df', 'data')
)
def footer_note(data_json):
    if not data_json:
        return "Tip: Start by uploading a CSV or Excel file."
    return "Use the KPI dropdown to drive all visualizations and scenario modeling. You can adjust the aggregation function for comparisons."


if __name__ == '__main__':
    app.run(debug=True)

# ----------------------------
# PDF Reporting (ReportLab)
# ----------------------------
def build_pdf_report(df: pd.DataFrame, kpi_col: str, date_col: str | None, multiplier: float, scenario_name: str, agg_func: str) -> bytes:
    """
    Build a lightweight PDF report summarizing dataset, KPI, and scenario settings.
    This intentionally avoids rasterizing interactive Plotly charts (keeps dependencies minimal).
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="DDDM Report",
        author="DDDM Tool",
    )
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Data-Driven Decision Making (DDDM) - Report", styles["Title"]))
    story.append(Spacer(1, 10))

    # Metadata
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta_rows = [
        ["Generated", gen_time],
        ["Rows", f"{len(df):,}"],
        ["Columns", f"{df.shape[1]:,}"],
        ["Selected KPI", kpi_col],
        ["Aggregation (category chart)", agg_func],
        ["Scenario", scenario_name],
        ["Scenario equation", f"{scenario_name} = baseline_mean({kpi_col}) x {multiplier:g}"],
    ]
    if date_col:
        try:
            dmin = pd.to_datetime(df[date_col], errors="coerce").min()
            dmax = pd.to_datetime(df[date_col], errors="coerce").max()
            if pd.notna(dmin) and pd.notna(dmax):
                meta_rows.append(["Date range", f"{dmin.date()} to {dmax.date()}"])
        except Exception:
            pass

    t = Table(meta_rows, colWidths=[6.0 * cm, 10.0 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 14))

    # KPI statistics
    story.append(Paragraph("KPI Summary Statistics", styles["Heading2"]))
    if kpi_col in df.columns:
        s = pd.to_numeric(df[kpi_col], errors="coerce")
        stats_rows = [
            ["Valid (non-null)", f"{s.notna().sum():,}"],
            ["Missing", f"{s.isna().sum():,}"],
            ["Mean", f"{s.mean():.4g}" if s.notna().any() else "N/A"],
            ["Median", f"{s.median():.4g}" if s.notna().any() else "N/A"],
            ["Std dev", f"{s.std():.4g}" if s.notna().any() else "N/A"],
            ["Min", f"{s.min():.4g}" if s.notna().any() else "N/A"],
            ["Max", f"{s.max():.4g}" if s.notna().any() else "N/A"],
        ]
        st = Table(stats_rows, colWidths=[6.0 * cm, 10.0 * cm])
        st.setStyle(
            TableStyle(
                [
                    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(st)
    else:
        story.append(Paragraph("No KPI selected or KPI not found in the dataset.", styles["BodyText"]))
    story.append(Spacer(1, 14))

    # Categorical overview (if any)
    story.append(Paragraph("Categorical Overview (Top Values)", styles["Heading2"]))
    cat_col = None
    for c in df.columns:
        if c == date_col:
            continue
        if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            nun = df[c].nunique(dropna=True)
            if 2 <= nun <= 20:
                cat_col = c
                break

    if cat_col:
        vc = df[cat_col].astype(str).value_counts(dropna=True).head(10)
        rows = [["Category column", cat_col]]
        rows += [[str(idx), f"{int(val):,}"] for idx, val in vc.items()]
        ct = Table(rows, colWidths=[10.0 * cm, 6.0 * cm])
        ct.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(ct)
    else:
        story.append(Paragraph("No suitable low-cardinality categorical column was detected (2-20 unique values).", styles["BodyText"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Note: Visualizations are interactive within the Dash application. This PDF summarizes key dataset and KPI information.", styles["Italic"]))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes





@app.callback(
    Output("download-pdf", "data"),
    Input("btn-download-pdf", "n_clicks"),
    State("stored-data", "data"),
    State("kpi-dropdown", "value"),
    State("date-dropdown", "value"),
    State("scenario-multiplier", "value"),
    State("scenario-name", "value"),
    State("agg-func", "value"),
    prevent_initial_call=True,
)
def download_pdf_report(n_clicks, stored, kpi_col, date_col, multiplier, scenario_name, agg_func):
    if not stored or not kpi_col:
        raise PreventUpdate

    df = pd.read_json(stored, orient="split")
    pdf_bytes = build_pdf_report(
        df=df,
        kpi_col=kpi_col,
        date_col=date_col,
        multiplier=float(multiplier) if multiplier is not None else 1.0,
        scenario_name=scenario_name or "Scenario",
        agg_func=agg_func or "mean",
    )
    filename = f"DDDM_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return dcc.send_bytes(lambda b: b.write(pdf_bytes), filename)
