import colorlover as cl
import plotly.graph_objs as go
import dash_html_components as html
import sklearn.metrics as metrics
import sklearn.calibration as calibration


def scale_score(array):
    return (array - array.min()) / (array.max() - array.min())


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


def serve_pr_curve(y_test, y_score, threshold=0.5):
    y_score = scale_score(y_score)
    y_pred = (y_score > threshold).astype(int)

    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        y_true=y_test,
        y_pred=y_pred,
        average='binary'
    )

    precision_array, recall_array, _ = metrics.precision_recall_curve(
        y_test,
        y_score
    )
    ap_score = metrics.average_precision_score(y_test, y_score)

    trace0 = go.Scatter(
        x=recall_array,
        y=precision_array,
        mode='lines',
        name='Predicted',
        fill='tozeroy',
        marker=dict(color='#F46036'),
    )

    data = [trace0]
    layout = go.Layout(
        title='Precision-Recall Curve: AP={0:0.2f}'.format(ap_score),
        margin=dict(t=35, b=35, r=30, l=50),
        xaxis=dict(title='Recall'),
        yaxis=dict(
            range=[0, 1.05],
            title='Precision'
        ),
        annotations=[
            dict(
                x=recall,
                y=precision,
                text='Threshold'
            )
        ]
    )

    figure = go.Figure(data=data, layout=layout)
    return figure


def serve_roc_curve(y_test, y_score, threshold=0.5):
    y_score = scale_score(y_score)
    y_pred = (y_score > threshold).astype(int)

    matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
    tn, fp, fn, tp = matrix.ravel()
    fpr = fp/(fp + tn)
    tpr = tp/(tp + fn)

    fpr_array, tpr_array, _ = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.roc_auc_score(y_test, y_score)

    trace0 = go.Scatter(
        x=fpr_array,
        y=tpr_array,
        mode='lines',
        name='Predicted',
        fill='tozeroy',
        marker=dict(color='#D7263D')
    )
    trace1 = serve_calibrated_line()

    data = [trace0, trace1]
    layout = go.Layout(
        title='ROC Curve: Area={0:0.2f}'.format(roc_auc),
        showlegend=False,
        margin=dict(t=35, b=35, r=30, l=50),
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(
            range=[0, 1.05],
            title='True Positive Rate'
        ),
        annotations=[
            dict(
                x=fpr,
                y=tpr,
                text='Threshold'
            )
        ]
    )

    figure = go.Figure(data=data, layout=layout)
    return figure


def serve_calibration_curve(y_test, y_score):
    y_score = scale_score(y_score)

    clf_score = metrics.brier_score_loss(y_test, y_score)
    # frac_pos --> fraction of positive
    # mpv --> mean predicted value
    frac_pos, mpv = calibration.calibration_curve(y_test, y_score, n_bins=10)

    trace0 = serve_calibrated_line()
    trace1 = go.Scatter(
        x=mpv,
        y=frac_pos,
        name='Predicted',
        mode='lines+markers',
        marker=dict(color='#2E294E'),
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
        yaxis='y2',
        marker=dict(color='#1B998B')
    )

    data = [trace0, trace1, trace2]
    layout = go.Layout(
        title='Calibration Plot: Brier={0:0.2f}'.format(clf_score),
        margin=dict(t=30, b=35, r=30, l=50),
        showlegend=False,
        yaxis=dict(
            title='Fraction of Positives',
            range=[0, 1.05],
            domain=[0.46, 1]
        ),
        xaxis2=dict(
            title='Mean Predicted Value',
            anchor='y2',
        ),
        yaxis2=dict(
            title='Count',
            domain=[0, 0.39]
        )
    )

    figure = go.Figure(data=data, layout=layout)
    return figure


def serve_pie_confusion_matrix(y_test, y_score, threshold=0.5):
    y_score = scale_score(y_score)

    y_pred = (y_score > threshold).astype(int)
    matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
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
        margin=dict(l=10, r=10, t=30, b=30),
        legend=dict(bgcolor='rgba(255,255,255,0)')
    )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    return figure


def serve_score_table(y_true, y_score, threshold=0.5):
    y_score = scale_score(y_score)

    y_pred = (y_score > threshold).astype(int)

    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        average='binary'
    )

    return html.Table(
        style={
            'margin-top': '20px',
            'width': '100%'
        },
        children=[
        html.Tr([
            html.Th(name) for name in ['Precision', 'Recall', 'F-Score']
        ]),

        html.Tr([
            html.Td(round(val, 4)) for val in [precision, recall, fscore]
        ])
    ])
