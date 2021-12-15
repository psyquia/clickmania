from django.urls import path
from .views import *

urlpatterns = [
    path('register', RegisterView.as_view()),
    path('login', LoginView.as_view()),
    path('update_user', UpdateUserView.as_view()),
    path('get_listings', GetSaleListingsView.as_view()),
    path('get_user', GetUserView.as_view()),
    path('buy_box', BuyLootBoxView.as_view()),
    path('get_user_arts', GetUserArtsView.as_view()),
    path('stop_selling', StopSellingView.as_view()),
    path('create_sale', CreateSaleView.as_view()),
    path('buy_from_market', BuyFromMarketView.as_view()),
]