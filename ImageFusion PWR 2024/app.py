from Presentation.MainPage import MainPage
from ConfigurationProvider import configurationProvider

app = MainPage(configurationProvider.GetConfiguration("MainPage"))
app.mainloop()