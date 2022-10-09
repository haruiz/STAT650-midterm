import dash_bootstrap_components as dbc

from dashboard_component import DashboardComponent


class NavBar(DashboardComponent):
    def __init__(self, app, title: str):
        super().__init__(app)
        self._title = title

    def layout(self):
        """
        Returns the layout for the navbar
        :return:
        """
        return dbc.NavbarSimple(
            id=str(id(self)),
            brand=self._title,
            color="dark",
            dark=True,
        )
