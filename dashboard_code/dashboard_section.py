import abc

from dash import Dash, html

from dashboard_component import DashboardComponent
from dataset import Dataset


class DashboardSection(DashboardComponent):
    def __init__(self, app: Dash, dataset: Dataset, title: str, description: str):
        """
        :param app: The Dash app
        :param app:
        :param dataset:
        :param title:
        :param description:
        """
        super().__init__(app)
        self.title = title
        self.description = description
        self.dataset = dataset

    @abc.abstractmethod
    def layout(self) -> html.Div:
        """
        Returns the layout for the section
        :return:
        """
        raise NotImplementedError("layout() not implemented")
