from django.contrib import admin

from .models import Collection, Art, User, Own, Sale

admin.site.register(Collection)
admin.site.register(Art)
admin.site.register(User)
admin.site.register(Own)
admin.site.register(Sale)

# Register your models here.
