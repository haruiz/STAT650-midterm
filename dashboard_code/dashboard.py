import dash_bootstrap_components as dbc
from dash import Dash, html

from dataset import Dataset
from navbar import NavBar
from sections import *


class Dashboard:
    def __init__(
        self,
        data_file: str = "data_final.csv",
        title: str = "Dashboard",
        description: str = "A dashboard",
    ):
        self._dataset = Dataset(data_file)
        self._title = title
        self._description = description
        # create the application
        self._app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
        self._app.title = title

    def _get_sections(self) -> list:
        """
        Returns a list of sections
        :return:
        """
        return [
            Q1Section(self._app, self._dataset),
            Q2Section(self._app, self._dataset),
            Q3Section(self._app, self._dataset),
            Q4Section(self._app, self._dataset),
        ]

    def build(self):
        """
        Builds the layout
        :return:
        """
        sections = list(
            map(
                lambda section: html.Div(
                    children=[
                        html.H1(children=section.title),
                        html.Div(children=section.description),
                        section.layout(),
                    ]
                ),
                self._get_sections(),
            )
        )
        # build the layout
        self._app.layout = dbc.Container(
            children=[
                NavBar(self._app, title="STAT650 - Midterm").layout(),
                html.H1(children=self._title),
                html.Div(children=self._description),
            ]
            + sections
        )

    def run(self):
        """
        Runs the application
        :return:
        """
        self.build()  # build the layout
        self._app.run_server(host='0.0.0.0')
