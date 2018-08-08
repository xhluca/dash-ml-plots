import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np

import figures

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
            html.H2('Binary Classification Dashboard'),
            html.Img(
                src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png")
        ])
    ]),

    html.Div(
        id='body',
        className='container scalable',
        style={'overflow-x': 'hidden'},
        children=[

            html.Div(
                className='four columns',
                style={'height': 'calc(100vh - 85px)'},
                children=[
                    dcc.Graph(
                        id='graph-calibration-curve',
                        figure=figures.serve_calibration_curve(y_test,
                                                               y_score),
                        style={'height': '99%'}
                    ),
                ]
            ),

            html.Div(
                className='four columns',
                style={'height': 'calc(100vh - 85px)'},
                children=[
                    dcc.Graph(
                        id='graph-roc-curve',
                        style={'height': '59%', 'padding-bottom': '2%'}
                    ),
                    dcc.Graph(
                        id='graph-confusion-matrix',
                        style={'height': '39%'}
                    ),

                ]
            ),

            html.Div(
                className='four columns',
                style={'height': 'calc(100vh - 85px)'},
                children=[
                    dcc.Graph(
                        id='graph-pr-curve',
                        style={'height': '59%', 'padding-bottom': '2%'}
                    ),

                    html.Div(style={'margin': '0px 20px 0px 50px'}, children=[
                        html.Div(id='div-score-table'),
                        html.Div(
                            id='div-current-threshold',
                            style={
                                'text-align': 'center',
                                'margin-bottom': '20px'
                            }
                        ),
                        dcc.Slider(
                            id='slider-threshold',
                            min=0,
                            max=1,
                            value=0.5,
                            step=0.01,
                            marks={
                                0: '0',
                                0.5: '0.5',
                                1: '1'
                            }
                        ),
                    ]),

                ]
            ),
        ]
    )
])


@app.callback(Output('div-current-threshold', 'children'),
              [Input('slider-threshold', 'value')])
def update_div_threshold(value):
    return "Current Threshold: " + str(value)


@app.callback(Output('graph-confusion-matrix', 'figure'),
              [Input('slider-threshold', 'value')])
def update_confusion_matrix(threshold):
    return figures.serve_pie_confusion_matrix(y_test, y_score, threshold)


@app.callback(Output('graph-pr-curve', 'figure'),
              [Input('slider-threshold', 'value')])
def update_pr_curve(threshold):
    return figures.serve_pr_curve(y_test, y_score, threshold)


@app.callback(Output('graph-roc-curve', 'figure'),
              [Input('slider-threshold', 'value')])
def update_pr_curve(threshold):
    return figures.serve_roc_curve(y_test, y_score, threshold)


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


@app.callback(Output('div-score-table', 'children'),
              [Input('slider-threshold', 'value')])
def update_score_table(threshold):
    return figures.serve_score_table(y_test, y_score, threshold)


for css in external_css:
    app.css.append_css({"external_url": css})

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)
