# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash(__name__)

data = pd.DataFrame(
    {
        "latitude": {
            0: 37.9844383,
            1: 37.987244200000006,
            2: 37.9783811,
            3: 37.9934691,
            4: 37.9842815,
            5: 37.9844383,
            6: 37.9844383,
            7: 39.29853170000001,
            8: 39.29853170000001,
            9: 39.29853170000001,
            10: 38.2280667,
            11: 38.2280667,
            12: 38.0045679,
            13: 37.9944925,
            14: 38.0158537,
            15: 38.0026161,
            16: 37.9823979,
            17: 37.9816306,
            18: 37.9787307,
            19: 37.9755042,
            20: 37.977031200000006,
            21: 37.9824516,
            22: 37.9756512,
            23: 37.9844383,
            24: 37.9755042,
            25: 37.9755042,
            26: 37.9779718,
            27: 37.9734542,
            28: 37.9769393,
            29: 40.6953398,
        },
        "longitude": {
            0: 23.7281172,
            1: 23.726373100000004,
            2: 23.7805126,
            3: 23.7275065,
            4: 23.7177254,
            5: 23.7281172,
            6: 23.7281172,
            7: 22.3844827,
            8: 22.3844827,
            9: 22.3844827,
            10: 21.7632893,
            11: 21.7632893,
            12: 23.7160906,
            13: 23.7042077,
            14: 23.7268584,
            15: 23.7334402,
            16: 23.6940037,
            17: 23.7281396,
            18: 23.7246486,
            19: 23.73267610000001,
            20: 23.7223758,
            21: 23.730563,
            22: 23.7340008,
            23: 23.7281172,
            24: 23.73267610000001,
            25: 23.73267610000001,
            26: 23.74299580000001,
            27: 23.735698,
            28: 23.743741600000003,
            29: 23.2203643,
        },
        "count": {
            0: 1,
            1: 3,
            2: 4,
            3: 4,
            4: 2,
            5: 4,
            6: 9,
            7: 4,
            8: 4,
            9: 7,
            10: 12,
            11: 9,
            12: 20,
            13: 5,
            14: 15,
            15: 12,
            16: 1,
            17: 4,
            18: 1,
            19: 2,
            20: 1,
            21: 1,
            22: 2,
            23: 3,
            24: 2,
            25: 2,
            26: 1,
            27: 2,
            28: 2,
            29: 1,
        },
    }
)

df = pd.DataFrame(
    {
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"],
    }
)

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
fig2 = px.density_mapbox(
    data, lat="latitude", lon="longitude", z="count", mapbox_style="stamen-terrain"
)
app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(
            children="""
        Dash: A web application framework for your data.
    """
        ),
        dcc.Graph(id="example-graph", figure=fig),
        dcc.Graph(id="example-graph2", figure=fig2),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)  # run the application in debug mode (for development)
