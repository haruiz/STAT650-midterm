import abc

from dash import Dash


class DashboardComponent(abc.ABC):
    def __init__(self, app: Dash):
        """
        :param app: The Dash app
        :param app:
        """
        self.app = app
        if hasattr(self, "callbacks"):
            self.callbacks(self.app)
