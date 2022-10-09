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
                lambda section: dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                html.H1(children=section.title),
                                html.Div(children=section.description),
                                section.layout(),
                            ],
                            md=12,
                        )
                    ]
                ),
                self._get_sections(),
            )
        )

        title_and_description = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H1(children=self._title),
                            html.Div(children=self._description),
                        ]
                    )
                )
            ]
        )

        group_members = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H3(children="Group Members"),
                            html.Div(
                                children="Alexander Peter, Henry Ruiz, Sabahat Zahra, Sai Manisha Duvvada, Sandeena Shrestha"
                            ),
                        ]
                    )
                )
            ]
        )

        github_repo = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H3(children="Github Repository"),
                            html.Div(
                                children=[
                                    html.A(
                                        href="https://github.com/haruiz/STAT650-midterm",
                                        children="STAT650-midterm",
                                    )
                                ]
                            ),
                        ]
                    )
                )
            ]
        )

        # build the layout
        self._app.layout = dbc.Container(
            children=[
                NavBar(self._app, title="STAT650 - Midterm").layout(),
                title_and_description,
                group_members,
                github_repo,
            ]
            + sections
        )

    def run(self):
        """
        Runs the application
        :return:
        """
        self.build()  # build the layout
        self._app.run_server(
            host="0.0.0.0"
        )  # run the application
