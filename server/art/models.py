from django.db import models
from django.templatetags.static import static
from django.contrib.auth.models import AbstractUser

class Collection(models.Model):
    collectionID = models.BigAutoField("collection ID", primary_key=True)
    name = models.CharField("name of collection", max_length=255, unique=True)
    display_name = models.CharField("display name of collection", max_length=255, default='')
    description = models.CharField("description of the collection", max_length=500, default='')

    @property
    def img_url(self):
        return static("artwork/{}/box/lootbox.png".format(self.name))

    def __str__(self):
        return self.name

class Art(models.Model):
    artID = models.BigAutoField("art ID", primary_key=True)
    title = models.CharField("title of artwork", max_length=255, unique=True)
    filename = models.CharField("filename", max_length=255, unique=False)
    rarity = models.IntegerField("rarity level of artwork", unique=False, default=1) # 1 is most common?
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, verbose_name="ID of collection art belongs to")

    def __str__(self):
        return self.title

    @property
    def img_url(self):
        return static("artwork/{}/{}".format(self.collection, self.filename))

    @property
    def thumb_url(self):
        return static("artwork/{}/thumb/{}".format(self.collection, self.filename))

class User(AbstractUser):
    email = models.EmailField("user email address", max_length=254, unique=True)
    coins = models.IntegerField("amount of currency user owns", unique=False, default=0)
    art = models.ManyToManyField(Art, through="Own", verbose_name="art owned by user")
    REQUIRED_FIELDS = ['email']

class Own(models.Model):
    ownID = models.BigAutoField("ownership instance ID", primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="ID of art owner")
    art = models.ForeignKey(Art, on_delete=models.CASCADE, verbose_name="ID of art")

    def __str__(self):
        return "{}-{}-{}".format(self.user, self.art, self.ownID)

    @property
    def duplicates(self):
        result = Own.objects.filter(user=self.user, art=self.art).count()
        return result

class Sale(models.Model):
    saleID = models.BigAutoField("sale ID", primary_key=True)
    seller = models.ForeignKey(User, on_delete=models.CASCADE, related_name="seller", verbose_name="ID of art seller")
    buyer = models.ForeignKey(User, default=None ,null=True, blank=True, on_delete=models.CASCADE, related_name="buyer", verbose_name="ID of art buyer")
    ownership = models.ForeignKey(Own, on_delete=models.CASCADE, verbose_name="ID of ownership instance")
    art = models.ForeignKey(Art, on_delete=models.CASCADE, verbose_name="ID of art")
    price = models.IntegerField("amount of currency seller requests", unique=False, default=10)
    available = models.BooleanField(default=True, verbose_name="is the product available for sale")
    sold = models.BooleanField(default=False, verbose_name="has the product been sold")
    postDate = models.DateTimeField(auto_now_add=True)
    purchaseDate = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return "Sale number {}".format(self.saleID)
