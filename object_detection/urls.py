from django.urls import path
from django.urls import path
from .views import home, register, user_login, user_logout, dashboard, process_image, add_image_feed, delete_image, image_detect

app_name = 'object_detection'

urlpatterns = [
    path('', home, name='home'),
    path('register/', register, name='register'),
    path('login/', user_login, name='login'),
    path('logout/', user_logout, name='logout'),
    path('dashboard/', dashboard, name='dashboard'),
    path('process_image/<int:feed_id>/', process_image, name='process_image'),
    path('add-image_feed/', add_image_feed, name='add_image_feed'),
    path('image/delete/<int:image_id>/', delete_image, name='delete_image'),
    path('image_detect/<int:pk>/', image_detect, name='image_detect'),
]