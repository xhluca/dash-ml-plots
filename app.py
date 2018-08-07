import os

import colorlover as cl
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics as skmetrics
import sklearn.calibration as skcalibration

app = dash.Dash(__name__)
server = app.server


def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Add noisy features
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # Limit to the two first classes, and split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
                                                        test_size=.5,
                                                        random_state=random_state)

    # Create a simple classifier
    classifier = svm.LinearSVC(random_state=random_state)
    classifier.fit(X_train, y_train)
    y_score = classifier.decision_function(X_test)
    y_score = \
        (y_score - y_score.min()) / (y_score.max() - y_score.min())

    return y_test, y_score


def serve_calibrated_line():
    return go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        hoverinfo='none',
        line=dict(
            color='#222222',
            dash='dash'
        )
    )


def serve_pr_curve(y_test, y_score):
    precision, recall, _ = skmetrics.precision_recall_curve(y_test, y_score)
    ap_score = skmetrics.average_precision_score(y_test, y_score)

    trace0 = go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name='Predicted',
        fill='tozeroy'
    )

    data = [trace0]
    layout = go.Layout(
        title='Precision-Recall Curve: AP={0:0.2f}'.format(ap_score),
        margin=dict(t=35, b=35, r=30, l=50),
        xaxis=dict(title='Recall'),
        yaxis=dict(title='Precision')
    )

    figure = go.Figure(data=data, layout=layout)
    return figure


def serve_roc_curve(y_test, y_score):
    fpr, tpr, _ = skmetrics.roc_curve(y_test, y_score)
    roc_auc = skmetrics.roc_auc_score(y_test, y_score)

    trace0 = go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name='Predicted',
        fill='tozeroy'
    )
    trace1 = serve_calibrated_line()

    data = [trace0, trace1]
    layout = go.Layout(
        title='ROC Curve: Area={0:0.2f}'.format(roc_auc),
        showlegend=False,
        margin=dict(t=35, b=35, r=30, l=50),
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate')
    )

    figure = go.Figure(data=data, layout=layout)
    return figure


def serve_calibration_curve(y_test, y_score):
    clf_score = skmetrics.brier_score_loss(y_test, y_score)
    # frac_pos --> fraction of positive
    # mpv --> mean predicted value
    frac_pos, mpv = skcalibration.calibration_curve(y_test, y_score, n_bins=10)

    trace0 = serve_calibrated_line()
    trace1 = go.Scatter(
        x=mpv,
        y=frac_pos,
        name='Predicted',
        mode='lines+markers'
    )
    trace2 = go.Histogram(
        x=frac_pos,
        xbins=dict(
            start=0,
            end=1,
            size=0.1
        ),
        opacity=0.8,
        xaxis='x2',
        yaxis='y2'
    )

    data = [trace0, trace1, trace2]
    layout = go.Layout(
        title='Calibration Plot: Brier={0:0.2f}'.format(clf_score),
        margin=dict(t=30, b=35, r=30, l=50),
        showlegend=False,
        xaxis=dict(

        ),
        yaxis=dict(
            title='Fraction of Positives',
            domain=[0.4, 1]
        ),
        xaxis2=dict(
            title='Mean Predicted Value',
            anchor='y2'
        ),
        yaxis2=dict(
            title='Count',
            domain=[0, 0.35]
        )
    )

    figure = go.Figure(data=data, layout=layout)
    return figure


def serve_pie_confusion_matrix(y_test, y_score):
    y_pred = (y_score > 0.5).astype(int)
    matrix = skmetrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
    tn, fp, fn, tp = matrix.ravel()

    values = [tp, fn, fp, tn]
    label_text = ["True Positive",
                  "False Negative",
                  "False Positive",
                  "True Negative"]
    labels = ["TP", "FN", "FP", "TN"]
    blue = cl.flipper()['seq']['9']['Blues']
    red = cl.flipper()['seq']['9']['Reds']
    colors = [blue[4], blue[1], red[1], red[4]]

    trace0 = go.Pie(
        labels=label_text,
        values=values,
        hoverinfo='label+value+percent',
        textinfo='text+value',
        text=labels,
        sort=False,
        marker=dict(
            colors=colors
        )
    )

    layout = go.Layout(
        title=f'Confusion Matrix',
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            bgcolor='rgba(255,255,255,0)',
            # orientation='h'
        )
    )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    return figure


y_test, y_score = load_data()

# Custom Script for Heroku
if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })

app.layout = html.Div(children=[
    # .container class is fixed, .container.scalable is scalable
    html.Div(className="banner", children=[
        html.Div(className='container scalable', children=[
            html.H2('App Name'),
            html.Img(
                src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png")
        ])
    ]),

    html.Div(id='body', className='container scalable', children=[

        html.Div(
            className='four columns',
            style={'height': '99vh'},
            children=[
                dcc.Graph(
                    id='graph-calibration-curve',
                    figure=serve_calibration_curve(y_test, y_score),
                    style={'height': '99%'}
                ),
            ]
        ),

        html.Div(
            className='four columns',
            style={'height': '99vh'},
            children=[
                dcc.Graph(
                    id='graph-pr-curve',
                    figure=serve_pr_curve(y_test, y_score),
                    style={'height': '49%', 'padding-bottom': '2%'}
                ),
                dcc.Graph(
                    id='graph-roc-curve',
                    figure=serve_roc_curve(y_test, y_score),
                    style={'height': '49%'}
                )
            ]
        ),

        html.Div(
            className='four columns',
            style={'height': '99vh'},
            children=[
                dcc.Graph(
                    id='graph-confusion-matrix',
                    figure=serve_pie_confusion_matrix(y_test, y_score),
                    style={'height': '48%', 'margin': '1%'}
                )
            ]
        ),
    ])
])

external_css = [
    # Normalize the CSS
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    # Fonts
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # Base Stylesheet, replace this with your own base-styles.css using Rawgit
    "https://cdn.rawgit.com/xhlulu/9a6e89f418ee40d02b637a429a876aa9/raw/base-styles.css",
    # Custom Stylesheet, replace this with your own custom-styles.css using Rawgit
    "https://cdn.rawgit.com/xhlulu/638e683e245ea751bca62fd427e385ab/raw/custom-styles.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)
